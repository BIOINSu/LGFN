# Local-Global Fusion Network for Video Super-Resolution

This repository is for LGFN. The source code will be released as soon as the paper is accepted.

## Introduction

The goal of video super-resolution technique is to effectively address the problem of restoring low-resolution (LR) videos to high-resolution (HR) ones. Traditional methods commonly use optical flow to perform frames alignment and design a framework from the perspective of space and time. However, inaccurate optical flow estimation may occur easily which leads to inferior quality restoration effects. In addition, how to effectively explore the dependence between video frames still remains a challenging problem. In this paper, we propose a Local-Global Fusion Network, termed LGFN, to solve the above issues from a novel viewpoint. As an alternative to optical flow, deformable convolution (DC) with a decreasing multi dilation convolution unit (DMDCU) is applied for implicit alignment efficiently. Moreover, a structure with two branches, consists of a Local Fusion Module (LFM) and a Global Fusion Module (GFM), is proposed to combine information from two different aspects. Specifically, LFM focuses on the relationship between adjacent frames and maintains the temporal consistency while GFM attempts to take advantages of all related features from an overall perspective. Benefit from our advanced network, experimental results on several datasets have demonstrated that our LGFM could not only achieve comparative performance with state-of-the-art methods but also possess reliable ability on restoring varieties of video frames. The results on benchmark datasets of our LGFN are presented on https://github.com/BIOINSu/LGFN and the source code will be released as soon as the paper is accepted.

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
