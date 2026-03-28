______________________________________________________________________
<div align="center">



# **_Documentation For Beginners_**

______________________________________________________________________

</div>

## Table of contents

- [About](#about)
- [0.) Getting Started](#0-getting-started)
- [1.) Segmentation](#1-segmentation)
- [2.) Vectorization](#2-vectorization)
- [3.) Geometry](#3-geometry)
- [4.) Quality](#4-quality)
- [5.) Encoding](#5-encoding)
- [6.) Configuring Custom Pipeline](#6-configuring-custom-pipeline)
- [7. Matching Entities](#7-matching-entities)

## About

This file will walk you through getting started with Worldcoin's Iris Recognition Inference System (IRIS). It connects to the internals files, documenting the inputs and outputs of each node along with visuals, and the three Colab files. It is recommended to go through the files in order, as they build upon each other. The files will give you an idea how the code works first, then help set up your own pipeline.

## 0. Getting Started

 **👉<a href="https://github.com/allison-diebol/open-iris/blob/main/colab/GettingStarted.ipynb">Getting Started Notebook</a>**

The Getting Started notebook is the first step in running your first `iris` code. This will walk you through you the basics of usage `iris` package. From it you will learn how to:
- Perform an `IRISPipeline` inference call.
- Configure `IRISPipeline` environment to modify error handling and return behaviour.
- Visualize `IRISPipeline` intermediate results.
- Download a sample image to use throughout the documentation files.
  - OR, use image capturing through your camera to get an image of your own eye!


## 1. Segmentation

 **👉<a href="https://github.com/allison-diebol/open-iris/blob/main/colab/internals/01_segmentation.ipynb">Segmentation Notebook</a>**

This Jupyter Notebook will walk you through the first step in iris recognition, `segmentation`, and does this through the `iris.nodes.segmentation` node. From it you will learn how to:
- Prepare an infrared (IR) eye image for processing.  
- Run the multilabel iris segmentation model.  
- Interpret the model’s pixel-wise class probability outputs.  
- Convert probabilities into a segmentation map.  
- Visualize the segmented iris regions.  
- Generate geometry and noise masks for downstream pipeline steps.

## 2. Vectorization

 **👉<a href="https://github.com/allison-diebol/open-iris/blob/main/colab/internals/02_vectorization.ipynb">Vectorization Notebook</a>**

This Jupyter Notebook will walk you through the second step in iris recognition, `vectorization`, and does this through the `iris.nodes.vectorization` node. From it you will learn how to:

- Initialize geometry refinement components for iris boundary processing.
- Apply contour interpolation to improve boundary continuity.
- Filter noisy contour points based on eyeball distance constraints.
- Smooth refined contours for more stable geometry representation.
- Use edge detection (Canny) to approximate iris boundaries.
- Prepare boundary information for downstream processing steps.

## 3. Geometry

 **👉<a href="https://github.com/allison-diebol/open-iris/blob/main/colab/internals/03_geometry.ipynb">Geometry Notebook</a>**

Next is the `geometry` step, and this is done through the nodes `iris.nodes.geometry_estimation` and `iris.nodes.eye_properties_estimation`. From it you will learn how to:

- Estimate the orientation of the iris using the Moment of Area method.
- Compute the center of the pupil and iris using the Bisectors Method.
- Extrapolate incomplete iris and pupil boundaries using Fusion Extrapolation.
- Understand the difference between circular (LinearExtrapolation) and elliptical (LSQEllipseFitWithRefinement) fitting methods.
- Handle occlusions by reconstructing missing parts of iris geometry.
- Validate that detected eye centers lie within acceptable image boundaries.
- Visualize geometric features such as orientation angles, center points, and reconstructed polygons.

## 4. Quality

 **👉<a href="https://github.com/allison-diebol/open-iris/blob/main/colab/internals/04_quality.ipynb">Quality Notebook</a>**

This Jupyter Notebook documents the `quality` assessment portion, covering `iris.nodes.eyes_properties_estimation`, `iris.nodes.aggregation`, and `iris.nodes.validators`. From it you will learn how to:

- Compute geometric properties of the pupil and iris using PupilIrisPropertyCalculator.
- Assess off-gaze via iris and pupil eccentricity with EccentricityOffgazeEstimation.
- Estimate the visible fraction of the iris (occlusion) with OcclusionCalculator.
- Aggregate multiple noise masks into a single unified mask using NoiseMaskUnion.
- Validate pupil-to-iris ratios, off-gaze, and iris visibility with Pupil2IrisPropertyValidator, OffgazeValidator, and OcclusionValidator.
- Visualize pupil/iris geometry, off-gaze scores, occlusion fractions, and pass/fail thresholds for each quality metric.

## 5. Encoding

 **👉<a href="https://github.com/allison-diebol/open-iris/blob/main/colab/internals/05_encoding.ipynb">Encoding Notebook</a>**

The final documentation notebook, `encoding`, will walk you through the iris recognition pipeline beyond segmentation, covering normalization, quality assessment, feature extraction, and encoding through:
- Converting the circular iris region into a normalized rectangular strip using LinearNormalization.
- Assessing image focus and sharpness with SharpnessEstimation and the SharpnessValidator.
- Extract complex Gabor filter responses from the normalized iris using ConvFilterBank.
- Identify and mask fragile/unreliable bits in the iris code with FragileBitRefinement.
- Encode the Gabor responses into a binary iris template using IrisEncoder and validate usable bits with IsMaskTooSmallValidator.
- Compute the bounding box of the iris in the original image with IrisBBoxCalculator.
- Visualize intermediate and final outputs, including normalized strips, noise masks, filter responses, refined masks, and binary iris codes.

## 6. Configuring Custom Pipeline

 **👉<a href="https://github.com/allison-diebol/open-iris/blob/main/colab/ConfiguringCustomPipeline.ipynb">Custom Pipeline Notebook</a>**

This Jupyter Notebook will walk you through you the steps you have to take to configure your custom `IRISPipeline`. From it you will learn how to:

- Configure `IRISPipeline` algorithms parameters.
- Configure `IRISPipeline` graph.
- Implement your own node with `Algorithm` class and introduce them into `IRISPipeline` graph.

## 7. Matching Entities

 **👉<a href="https://github.com/allison-diebol/open-iris/blob/main/colab/MatchingEntities.ipynb">Matching Notebook</a>**

This Jupyter Notebook will walk you through the basics of how to use matchers available in the `iris` package. From it you will learn how to:

- Use the `HammingDistanceMatcher` matcher to compute distance between two eyes.