# Local-Global Fusion Network for Video Super-Resolution

This repository is about Local-Global Fusioin Network for Video Super-Resolution (LGFN). The source code will be released as soon as the paper is accepted.

## Introduction

Our proposed LGFN devotes to effectively address the problem of restoring low-resolution (LR) videos to high-resolution (HR) ones. The input frames are first extract features by residual blocks. Next, we align the neighboring frames with the reference frame through stacked deformable convolutions (DCs) with decreased multi dilation convolution unit (DMDCU) to predict more accurate parameters. After that, features are fused by Local Fusion Module and Global Fusion Module respectively. The outputs are integrated together and sent into the reconstruction module to restore high quality video frames.

## Experimental Results

### Quantitative Results

Quantitative results of state-of-the-art SR algorithms on Vid4 for 4 ×.  **Bold type** indicates the best and <u>underline</u> indicates the second best performance (PSNR/SSIM). 

![quantitative_vid4](./imgs/quantitative_vid4.png)



Quantitative results of state-of-the-art SR algorithms on SPMCS for 4 ×. The results are the average evaluation across all restored video frames.  **Bold type** indicates the best and <u>underline</u> indicates the second best performance (PSNR/SSIM). 

![quantitative_spmcs_all](./imgs/quantitative_spmcs_all.png)



Quantitative results of state-of-the-art SR algorithms on part of SPMCS clips for 4 ×.  **Bold type** indicates the best and <u>underline</u> indicates the second best performance (PSNR/SSIM). 

![quantitative_spmcs_part](./imgs/quantitative_spmcs_part.png)



Quantitative results of state-of-the-art SR algorithms on Vimeo-90K-T for 4 ×.  **Bold type** indicates the best and <u>underline</u> indicates the second best performance (PSNR/SSIM). 

![quantitative_vimeo90k](./imgs/quantitative_vimeo90k.png)



### Visual Results

Visual comparisons on Vid4 for 4 × scaling factor. Zoom in to see better visualization.

![visual_spmcs](./imgs/visual_spmcs.png)



Visual comparisons on SPMCS for 4 × scaling factor.

![visual_spmcs](./imgs/visual_spmcs.png)



Visual comparisons on Vimeo-90K-T for 4 × scaling factor.

![visual_vimeo90k](./imgs/visual_vimeo90k.png)
