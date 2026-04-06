
## Benchmarking
Contains information and testing related to how the iris package performs on recognition tasks using publicly available datasets like CASIA Iris Thousand, the IIT Delhi Iris Database, and the Hong Kong Polytechnic University Cross-Spectral Iris Images Database. 

**CASIA_IITD_Benchmark.ipynb**: compares CASIA and IITD databases on metrics like recognition accuracy using a 0.27 Hamming Distance (HD) threshold, mean genuine HD, and mean impostor HD

**Band-comparison.ipynb**: tests how the iris package, which is designed for IR input images, performs on greyscale visual wavelength images and on red-channel-only visual wavelength images. 

**Downsampling-comparison.ipynb**: tests how pre-segmentation downsampling of input images affects IrisPipeline speed and accuracy.


