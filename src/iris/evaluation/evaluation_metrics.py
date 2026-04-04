"""
iris_evaluation.py
------------------
Built-in evaluation suite for open-iris pipeline.

Provides:
  - compute_match_scores()    batch-compute Hamming distances for genuine/impostor pairs
  - compute_roc()             ROC curve (FPR, TPR, thresholds)
  - compute_far_frr()         FAR / FRR at every threshold
  - compute_eer()             Equal Error Rate + operating threshold
  - evaluate_dataset()        end-to-end evaluation from a folder of images
  - plot_roc()                matplotlib ROC curve
  - plot_det()                Detection Error Tradeoff (DET) curve
  - plot_far_frr()            FAR/FRR vs threshold (shows EER crossing)
  - evaluation_report()       prints a human-readable summary

Requirements
------------
  pip install open-iris matplotlib scipy numpy
"""

from __future__ import annotations

import itertools
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── lazy imports for plotting (optional) ────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    _MATPLOTLIB = True
except ImportError:
    _MATPLOTLIB = False

try:
    from scipy.special import ndtri  # for DET curve (probit scale)
    _SCIPY = True
except ImportError:
    _SCIPY = False

import iris
from iris.io.dataclasses import IrisTemplate

@dataclass
class MatchPair:
    """One pairwise comparison result."""
    id_a: str
    id_b: str
    score: float          # fractional Hamming distance  (0 = identical)
    is_genuine: bool      # True  → same subject  |  False → different subjects

@dataclass
class EvaluationResult:
    """Aggregated evaluation outputs."""
    genuine_scores: np.ndarray
    impostor_scores: np.ndarray
    thresholds: np.ndarray
    far: np.ndarray
    frr: np.ndarray
    fpr: np.ndarray       # alias for FAR (used in ROC)
    tpr: np.ndarray       # 1 - FRR
    eer: float
    eer_threshold: float
    auc: float
    pairs: List[MatchPair] = field(default_factory=list)

def _hamming_distance(t1: IrisTemplate, t2: IrisTemplate) -> float:
    """
    Fractional Hamming distance between two IrisTemplates.

    Uses the mask-aware formula:
        HD = (A XOR B) AND (mask_A AND mask_B)  /  |mask_A AND mask_B|

    Averaged across all filter bands.
    """
    total_bits, mismatch_bits = 0, 0
    for ic1, mc1, ic2, mc2 in zip(t1.iris_codes, t1.mask_codes, t2.iris_codes, t2.mask_codes):
        combined_mask = mc1 & mc2
        n = combined_mask.sum()
        if n == 0:
            continue
        xor = (ic1 ^ ic2) & combined_mask
        mismatch_bits += xor.sum()
        total_bits += n
    return float(mismatch_bits) / float(total_bits) if total_bits > 0 else 1.0

def compute_match_scores(templates: Dict[str, IrisTemplate], pairs: List[Tuple[str, str, bool]],) -> List[MatchPair]:
    """
    Compute Hamming distances for a list of (id_a, id_b, is_genuine) pairs.

    Parameters
    ----------
    templates : dict mapping image_id → IrisTemplate
    pairs     : list of (id_a, id_b, is_genuine)

    Returns
    -------
    List[MatchPair]
    """
    results = []
    for id_a, id_b, genuine in pairs:
        if id_a not in templates or id_b not in templates:
            continue
        dist = _hamming_distance(templates[id_a], templates[id_b])
        results.append(MatchPair(id_a=id_a, id_b=id_b, score=dist, is_genuine=genuine))
    return results

def compute_far_frr(match_pairs: List[MatchPair], n_thresholds: int = 500,) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute FAR, FRR across a sweep of thresholds.

    A *match* is declared when score <= threshold.

    Returns
    -------
    thresholds : array of shape (n_thresholds,)
    far        : False Accept Rate at each threshold
    frr        : False Reject Rate at each threshold
    """
    genuine_scores  = np.array([p.score for p in match_pairs if p.is_genuine])
    impostor_scores = np.array([p.score for p in match_pairs if not p.is_genuine])

    all_scores = np.concatenate([genuine_scores, impostor_scores])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), n_thresholds)

    far, frr = np.zeros(n_thresholds), np.zeros(n_thresholds)

    n_genuine  = len(genuine_scores)
    n_impostor = len(impostor_scores)

    for i, t in enumerate(thresholds):
        # FAR: impostors incorrectly accepted (score <= threshold)
        fa = (impostor_scores <= t).sum()
        # FRR: genuines incorrectly rejected (score > threshold)
        fr = (genuine_scores > t).sum()

        far[i] = fa / n_impostor if n_impostor > 0 else 0.0
        frr[i] = fr / n_genuine  if n_genuine  > 0 else 0.0

    return thresholds, far, frr


def compute_eer(thresholds: np.ndarray, far: np.ndarray, frr: np.ndarray,) -> Tuple[float, float]:
    """
    Equal Error Rate: the threshold where FAR ≈ FRR.

    Returns
    -------
    eer           : float  (value between 0 and 1)
    eer_threshold : float
    """
    diff = np.abs(far - frr)
    idx  = diff.argmin()
    eer  = float((far[idx] + frr[idx]) / 2)
    return eer, float(thresholds[idx])


def compute_roc(thresholds: np.ndarray, far: np.ndarray, frr: np.ndarray,) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    ROC curve: (FPR, TPR) and AUC via trapezoidal integration.

    Returns
    -------
    fpr : same as far
    tpr : 1 - frr
    auc : float
    """
    fpr = far
    tpr = 1.0 - frr
    # ensure monotone for AUC
    order = np.argsort(fpr)
    fpr_s, tpr_s = fpr[order], tpr[order]
    auc = float(np.trapz(tpr_s, fpr_s))
    return fpr, tpr, auc

def _subject_id_from_filename(filename: str) -> str:
    """
    Infer subject ID from filename.

    Supports common naming conventions:
      001_L_01.png  →  "001"
      subject002_session1.jpg  →  "subject002"
      s05_eye1.bmp  →  "s05"

    Falls back to the full stem if no pattern matches.
    """
    stem = Path(filename).stem
    # try leading digits / alphanumeric token before first underscore or dash
    m = re.match(r'^([A-Za-z]*\d+)', stem)
    return m.group(1) if m else stem

def load_templates_from_folder(folder: str | Path, eye_side: str = "left", pipeline: Optional[iris.IRISPipeline] = None, verbose: bool = True,) -> Dict[str, IrisTemplate]:
    """
    Run IRISPipeline over every image in *folder* and return a dict of
    image_id → IrisTemplate. Failed images are skipped.

    Now prints progress every 100 images instead of per image.
    """
    folder = Path(folder)

    if pipeline is None:
        pipeline = iris.IRISPipeline()

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    image_files = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
    templates: Dict[str, IrisTemplate] = {}
    ok_count = 0
    skip_count = 0
    total = len(image_files)

    for i, img_path in enumerate(image_files, 1):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            skip_count += 1
            continue

        output = pipeline(
            iris.IRImage(
                img_data=img,
                image_id=img_path.stem,
                eye_side=eye_side
            )
        )

        if output.get("error") is not None:
            skip_count += 1
            continue

        templates[img_path.stem] = output["iris_template"]
        ok_count += 1

        if verbose and i % 100 == 0:
            pct = (i / total) * 100
            print(f"[{pct:.1f}%] Processed {i}/{total} → OK: {ok_count}, SKIP: {skip_count}")

    if verbose:
        print(f"\nFinished processing {total} images")
        print(f"OK: {ok_count}")
        print(f"SKIP: {skip_count}")

    return templates


def build_pairs_by_subject(image_ids: List[str], subject_fn=_subject_id_from_filename,) -> List[Tuple[str, str, bool]]:
    """
    Build all pairwise combinations and label them genuine/impostor
    using *subject_fn* to map image_id → subject_id.

    Returns
    -------
    List of (id_a, id_b, is_genuine)
    """
    pairs = []
    for a, b in itertools.combinations(image_ids, 2):
        genuine = subject_fn(a) == subject_fn(b)
        pairs.append((a, b, genuine))
    return pairs


def evaluate_dataset(folder: str | Path, eye_side: str = "left", n_thresholds: int = 500, subject_fn=_subject_id_from_filename, verbose: bool = True,) -> EvaluationResult:
    """
    Full end-to-end evaluation from a folder of eye images.

    Steps
    -----
    1. Load images → extract IrisTemplates
    2. Build all pairwise genuine/impostor comparisons
    3. Compute match scores (Hamming distances)
    4. Compute FAR, FRR, EER, ROC, AUC

    Parameters
    ----------
    folder        : path to folder of IR eye images
    eye_side      : "left" or "right"
    n_thresholds  : resolution of FAR/FRR sweep
    subject_fn    : callable(image_id) → subject_id

    Returns
    -------
    EvaluationResult
    """
    print(f"\nLoading templates from: {folder}")
    templates = load_templates_from_folder(folder, eye_side=eye_side, verbose=verbose)

    if len(templates) < 2:
        raise ValueError("Need at least 2 successfully processed images.")

    print(f"\nBuilding pairs from {len(templates)} templates …")
    pairs_spec = build_pairs_by_subject(list(templates.keys()), subject_fn)
    print(f"    Genuine pairs : {sum(1 for _,_,g in pairs_spec if g)}")
    print(f"    Impostor pairs: {sum(1 for _,_,g in pairs_spec if not g)}")

    print("\nComputing match scores …")
    match_pairs = compute_match_scores(templates, pairs_spec)

    thresholds, far, frr = compute_far_frr(match_pairs, n_thresholds)
    eer, eer_threshold    = compute_eer(thresholds, far, frr)
    fpr, tpr, auc         = compute_roc(thresholds, far, frr)

    genuine_scores  = np.array([p.score for p in match_pairs if p.is_genuine])
    impostor_scores = np.array([p.score for p in match_pairs if not p.is_genuine])

    return EvaluationResult(
        genuine_scores=genuine_scores,
        impostor_scores=impostor_scores,
        thresholds=thresholds,
        far=far,
        frr=frr,
        fpr=fpr,
        tpr=tpr,
        eer=eer,
        eer_threshold=eer_threshold,
        auc=auc,
        pairs=match_pairs,
    )

def _require_matplotlib():
    if not _MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting: pip install matplotlib")


def plot_roc(result: EvaluationResult, save_path: Optional[str] = None) -> None:
    """Plot ROC curve (FPR vs TPR)."""
    _require_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(result.fpr, result.tpr, lw=2, color="#2563eb",
            label=f"IRIS  (AUC = {result.auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", lw=1)
    ax.set_xlabel("False Accept Rate (FAR / FPR)")
    ax.set_ylabel("Genuine Accept Rate (1 - FRR / TPR)")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved → {save_path}")
    else:
        plt.show()


def plot_det(result: EvaluationResult, save_path: Optional[str] = None) -> None:
    """
    Detection Error Tradeoff (DET) curve on probit/normal deviate scale.
    Falls back to linear scale when scipy is unavailable.
    """
    _require_matplotlib()
    far, frr = result.far, result.frr

    fig, ax = plt.subplots(figsize=(6, 6))
    if _SCIPY:
        # clip to avoid infinity at 0/1
        eps = 1e-6
        x = ndtri(np.clip(far, eps, 1 - eps))
        y = ndtri(np.clip(frr, eps, 1 - eps))
        ax.set_xlabel("FAR (normal deviate scale)")
        ax.set_ylabel("FRR (normal deviate scale)")
    else:
        x, y = far, frr
        ax.set_xlabel("FAR")
        ax.set_ylabel("FRR")

    ax.plot(x, y, lw=2, color="#7c3aed")
    # EER point
    eer_x = ndtri(np.clip(result.eer, 1e-6, 1 - 1e-6)) if _SCIPY else result.eer
    ax.scatter([eer_x], [eer_x], color="#dc2626", zorder=5,
               label=f"EER = {result.eer*100:.2f}%")
    ax.set_title("DET Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved → {save_path}")
    else:
        plt.show()


def plot_far_frr(result: EvaluationResult, save_path: Optional[str] = None) -> None:
    """Plot FAR and FRR vs threshold, with EER crossing highlighted."""
    _require_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result.thresholds, result.far * 100, lw=2, color="#2563eb", label="FAR")
    ax.plot(result.thresholds, result.frr * 100, lw=2, color="#dc2626", label="FRR")
    ax.axvline(result.eer_threshold, color="#16a34a", lw=1.5, linestyle="--",
               label=f"EER threshold = {result.eer_threshold:.4f}")
    ax.axhline(result.eer * 100, color="#f59e0b", lw=1.5, linestyle=":",
               label=f"EER = {result.eer*100:.2f}%")
    ax.set_xlabel("Decision Threshold (Hamming Distance)")
    ax.set_ylabel("Error Rate (%)")
    ax.set_title("FAR / FRR vs Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved → {save_path}")
    else:
        plt.show()

def plot_score_distributions(result: EvaluationResult, save_path: Optional[str] = None, bins: int = 50,) -> None:
    """Overlay histograms of genuine and impostor Hamming distances."""
    _require_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(result.genuine_scores,  bins=bins, alpha=0.6, color="#2563eb",
            label=f"Genuine  (n={len(result.genuine_scores)})",  density=True)
    ax.hist(result.impostor_scores, bins=bins, alpha=0.6, color="#dc2626",
            label=f"Impostor (n={len(result.impostor_scores)})", density=True)
    ax.axvline(result.eer_threshold, color="#16a34a", lw=1.5, linestyle="--",
               label=f"EER @ {result.eer_threshold:.4f}")
    ax.set_xlabel("Hamming Distance")
    ax.set_ylabel("Density")
    ax.set_title("Genuine vs Impostor Score Distributions")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved → {save_path}")
    else:
        plt.show()

def evaluation_report(result: EvaluationResult) -> str:
    """Return a human-readable evaluation summary string."""
    # FAR/FRR at common operating points
    lines = [
        "═" * 52,
        "  open-iris  |  Evaluation Report",
        "═" * 52,
        f"  Genuine comparisons : {len(result.genuine_scores):,}",
        f"  Impostor comparisons: {len(result.impostor_scores):,}",
        "",
        f"  EER                 : {result.eer*100:.3f} %",
        f"  EER threshold       : {result.eer_threshold:.5f}",
        f"  AUC (ROC)           : {result.auc:.5f}",
        "",
        "  Operating points:",
    ]

    for target_far in [0.001, 0.01, 0.1]:
        idx = np.argmin(np.abs(result.far - target_far))
        lines.append(
            f"    FAR={target_far*100:.1f}%  →  "
            f"FRR={result.frr[idx]*100:.2f}%  "
            f"(threshold={result.thresholds[idx]:.4f})"
        )

    lines.append("═" * 52)
    report = "\n".join(lines)
    print(report)
    return report