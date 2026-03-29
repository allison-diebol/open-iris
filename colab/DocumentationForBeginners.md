______________________________________________________________________
<div align="center">



# **_Documentation For Beginners_**

______________________________________________________________________

</div>

## Table of contents

- [About](#about)
- [Getting Started](#getting-started)
- [Segmentation](#segmentation)
- [Vectorization](#vectorization)
- [Geometry](#geometry)
- [Quality](#quality)
- [Encoding](#encoding)
- [Configuring Custom Pipeline](#configuring-custom-pipeline)
- [Matching Entities](#matching-entities)
- [Common Issues](#common-issues)
- [Try It Yourself: Camera Implementation](#try-it-yourself-camera-implementation)

## About

This file will walk you through getting started with Worldcoin's Iris Recognition Inference System (IRIS). It connects to the internals files, developed by the Michigan State University Data Science Capstone Team, and the three Colab files, devloped by Worldcoin. It is recommended to go through the files in order, as they build upon each other. The files will give you an idea how the code works first, then help set up your own pipeline. Before walking through these notebooks, please ensure that you have the proper installation, as described in the repository's `README` file.

This guide is intended for:
- Developers new to iris recognition
- Users of the `iris` package
- Researchers exploring biometric pipelines.

## Getting Started

 **<a href="https://github.com/allison-diebol/open-iris/blob/main/colab/GettingStarted.ipynb">Getting Started Notebook</a>**

The Getting Started notebook is the first step in running your first `iris` code. This will walk you through the basics of usage `iris` package. From it you will learn how to:
- Perform an `IRISPipeline` inference call.
- Configure `IRISPipeline` environment to modify error handling and return behaviour.
- Visualize `IRISPipeline` intermediate results.
- Download a sample image to use throughout the documentation files.
  - OR, use image capturing through your camera to get an image of your own eye!


## Segmentation

 **<a href="https://github.com/allison-diebol/open-iris/blob/main/colab/internals/01_segmentation.ipynb">Segmentation Notebook</a>**

Segmentation focuses on isolating the iris region. This Jupyter Notebook will walk you through the first step in iris recognition, `segmentation`, and does this through the `iris.nodes.segmentation` node. From it you will learn how to:
- Prepare an infrared (IR) eye image for processing.  
- Run the multilabel iris segmentation model.  
- Interpret the model’s pixel-wise class probability outputs.  
- Convert probabilities into a segmentation map.  
- Visualize the segmented iris regions.  
- Generate geometry and noise masks for downstream pipeline steps.

Classes Covered: `MultilabelSegmentation`, `MultilabelSegmentationBinarization`

## Vectorization

 **<a href="https://github.com/allison-diebol/open-iris/blob/main/colab/internals/02_vectorization.ipynb">Vectorization Notebook</a>**

Vectorization focuses on refining the boundaries of the image and iris. This Jupyter Notebook will walk you through the second step in iris recognition, `vectorization`, and does this through the `iris.nodes.vectorization` node. From it you will learn how to:

- Initialize geometry refinement components for iris boundary processing.
- Apply contour interpolation to improve boundary continuity.
- Filter noisy contour points based on eyeball distance constraints.
- Smooth refined contours for more stable geometry representation.
- Use edge detection (Canny) to approximate iris boundaries.
- Prepare boundary information for downstream processing steps.

Classes Covered: `ContouringAlgorithm`, `ContourPointNoiseEyeballDistanceFilter`, `Smoothing`

## Geometry

 **<a href="https://github.com/allison-diebol/open-iris/blob/main/colab/internals/03_geometry.ipynb">Geometry Notebook</a>**

Next is the `geometry` step, and this is done through the nodes `iris.nodes.geometry_estimation` and `iris.nodes.eye_properties_estimation`. From it you will learn how to:

- Estimate the orientation of the iris using the Moment of Area method.
- Compute the center of the pupil and iris using the Bisectors Method.
- Extrapolate incomplete iris and pupil boundaries using Fusion Extrapolation.
- Understand the difference between circular (LinearExtrapolation) and elliptical (LSQEllipseFitWithRefinement) fitting methods.
- Handle occlusions by reconstructing missing parts of iris geometry.
- Validate that detected eye centers lie within acceptable image boundaries.
- Visualize geometric features such as orientation angles, center points, and reconstructed polygons.

Classes Covered: `MomentOfArea`, `BisectorsMethod`, `FusionExtrapolation`, `LinearExtrapolation`, `LSQEllipseFitWithRefinement`, `LinearExtrapolation`, `EyeCentersInsideImageValidator`


## Quality

 **<a href="https://github.com/allison-diebol/open-iris/blob/main/colab/internals/04_quality.ipynb">Quality Notebook</a>**

This Jupyter Notebook documents the `quality` assessment portion, covering `iris.nodes.eyes_properties_estimation`, `iris.nodes.aggregation`, and `iris.nodes.validators`. From it you will learn how to:

- Compute geometric properties of the pupil and iris using PupilIrisPropertyCalculator.
- Assess off-gaze via iris and pupil eccentricity with EccentricityOffgazeEstimation.
- Estimate the visible fraction of the iris (occlusion) with OcclusionCalculator.
- Aggregate multiple noise masks into a single unified mask using NoiseMaskUnion.
- Validate pupil-to-iris ratios, off-gaze, and iris visibility with Pupil2IrisPropertyValidator, OffgazeValidator, and OcclusionValidator.
- Visualize pupil/iris geometry, off-gaze scores, occlusion fractions, and pass/fail thresholds for each quality metric.

Classes Covered: `PupilIrisPropertyCalculator`, `EccentricityOffgazeEstimation`, `OcclusionCalculator`, `NoiseMaskUnion`, `Pupil2IrisPropertyValidator`, `OffgazeValidator`, `OcclusionValidator`


## Encoding

 **<a href="https://github.com/allison-diebol/open-iris/blob/main/colab/internals/05_encoding.ipynb">Encoding Notebook</a>**

The final documentation notebook, `encoding`, will walk you through the iris recognition pipeline beyond segmentation, covering normalization, quality assessment, feature extraction, and encoding through:
- Converting the circular iris region into a normalized rectangular strip using LinearNormalization.
- Assessing image focus and sharpness with SharpnessEstimation and the SharpnessValidator.
- Extract complex Gabor filter responses from the normalized iris using ConvFilterBank.
- Identify and mask fragile/unreliable bits in the iris code with FragileBitRefinement.
- Encode the Gabor responses into a binary iris template using IrisEncoder and validate usable bits with IsMaskTooSmallValidator.
- Compute the bounding box of the iris in the original image with IrisBBoxCalculator.
- Visualize intermediate and final outputs, including normalized strips, noise masks, filter responses, refined masks, and binary iris codes.

Classes Covered: `LinearNormalization`, `SharpnessEstimation`, `SharpnessValidator`, `ConvFilterBank`, `GaborFilter`, `RegularProbeSchema`, `FragileBitRefinement`, `IrisEncoder`, `IsMaskTooSmallValidator`, `IrisBBoxCalculator`

## Configuring Custom Pipeline

 **<a href="https://github.com/allison-diebol/open-iris/blob/main/colab/ConfiguringCustomPipeline.ipynb">Custom Pipeline Notebook</a>**

This Jupyter Notebook will walk you through you the steps you have to take to configure your custom `IRISPipeline`. From it you will learn how to:

- Configure `IRISPipeline` algorithms parameters.
- Configure `IRISPipeline` graph.
- Implement your own node with `Algorithm` class and introduce them into `IRISPipeline` graph.

## Matching Entities

 **<a href="https://github.com/allison-diebol/open-iris/blob/main/colab/MatchingEntities.ipynb">Matching Notebook</a>**

This Jupyter Notebook will walk you through the basics of how to use matchers available in the `iris` package. From it you will learn how to:

- Use the `HammingDistanceMatcher` matcher to compute distance between two eyes.

## Common Issues

Iris recognition depends heavily on image quality. Poor focus, noise, motion blur, or obstructions like eyelids or eyelashes can significantly lower performance. 
- Poor lighting can potentially result in segmentation failure
- Occlusions (eyelids, eyelashes) can lead to reduced quality scores
- Low resolution can result in unreliable encoding

## Try It Yourself: Camera Implementation
