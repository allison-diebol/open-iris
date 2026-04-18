
## Benchmarking Information

This folder contains notebooks for benchmarking the **iris** package on recognition tasks. We evaluate performance across several publicly available datasets and explore trade-offs between accuracy, robustness, and speed.

### Datasets

There are many public iris datasets available online yet most require some form of application to gain access to. We use the following datasets in our evaluation:

- *CASIA Iris Thousand*
  
  A large-scale iris dataset with 20,000+ images from approximately 1,000 subjects, commonly used for benchmarking recognition performance. They are captured in near-infrared (NIR) and are designed to introduce real-world variability. There is more intra-class variation including occlusions, slight motion blur, and illumination variation for testing scalability and robustness.
  
  [Official Iris Thousand](https://hycasia.github.io/dataset/casia-irisv4/) and [Kaggle Mirror](https://www.kaggle.com/datasets/sondosaabed/casia-iris-thousand)

- *IIT Delhi Iris Database*
  
  A controlled dataset of approximately 2,200 images from about 224 subjects collected in near-infrared (NIR), widely used in academic research. This set's images were collected in a controlled, indoor environment leading to high-quality, low-noise images with minimal occlusion and blur. It often produces optimistic accuracy results and is thus less representative of real-world deployment conditions.
  
  [IITD Database Access](https://www4.comp.polyu.edu.hk/~csajaykr/IITD/Database_Iris.htm)

- *Hong Kong Polytechnic University Cross-Spectral Iris Database*
  
  Contains 12,540 images in both visible-light and NIR wavelengths, useful for testing cross-spectral robustness due to its simultaneous bi-spectral imaging.
  
  [Hong Kong PolyU Access](https://www4.comp.polyu.edu.hk/~csajaykr/polyuiris.htm)

### Notebooks
[`01_iris_pipeline_encoding.ipynb`](01_iris_pipeline_encoding.ipynb)

Handles dataset ingestion and pipeline execution:
- Builds the structured image dataset used throughout benchmarking
- Encodes iris templates
- Logs failures and pipeline outputs

[`02_iris_failure_analysis.ipynb`](02_iris_failure_analysis.ipynb)

Analyzes pipeline failures to understand breakdown points and improvement opportunities:
- Identifies common error modes
- Investigates challenging samples
- Evaluates preprocessing-based recovery strategies
- Explores segmentation vs. vectorization failure attribution
  
[`CASIA_IITD_Benchmark.ipynb`](CASIA_IITD_Benchmark.ipynb)

Benchmark comparison between CASIA and IITD datasets. Includes:
- Recognition accuracy at 0.27 Hamming Distance (HD) threshold
- Mean genuine HD
- Mean imposter HD

[`band-comparison.ipynb`](band-comparison.ipynb)

Test how the pipeline, originally designed for IR images, performs on visible-spectrum inputs, analyzing results for:
- near-infrared (NIR)
- visual-light (VIS)
- red-channel of visual-light (VIS_R)
alongside looking into balanced and unbalanced sets as well as same-session and cross-session experiments. 

[`downsampling-comparison.ipynb`](downsampling-comparison.ipynb)

Explores how downsampling images before segmentation impacts:
- Pipeline runtime
- Recognition accuracy
