###### tags: `ccClub2022fall`

# Landscape Painter(github README.md)


> We provide a web application to transfer the image from a segmentation map to a photorealistic image. In the application, you can choose the  SPADE or the PsP encoder with StyleGANv2 to generate your painting. We also provide the docker images and the checkpoint files that can quickly build the application on your own. 

<p align="center">
<img src="posters/demo.gif" width="800px"/>
</p>
<div style='text-align: center'>By default, we only provid 5 classes to draw. You can reference the <a href="https://groups.csail.mit.edu/vision/datasets/ADE20K/">ADE20K</a> document if you want to change the class.<p><b>(Note that the drawing color is equal to the label color in ADE20K)</b></div>

## Table of Contents
* [Goals](##Goals)
* [Architecture](##Architecture)
* [Datasets](##Datasets)
* [Getting Started](##Getting-Started)
    * [Prerequisites](#Prerequisites)
    * [Installation](#Installation)
* [Demo](##Demo)
* [Demo](##Demo)
* [Demo](##Demo)


## Goals
1. Develop a web application
2. 讓使用者可以透過色塊輕鬆繪製真實場景
3. 建立易重建的環境(based on docker)

## Architecture
![](https://i.imgur.com/zvaTAqU.png)


## Getting Started
### Prerequisites
* Computer(Linux or Windows is fine, but it doesn't support Mac)
* Nvidia GPU, driver need to be installed. (The tutorial in [Datasets](##Datasets) may help you)
* For Windows user, WSL2 need to be installed. (The tutorial in [Datasets](##Datasets) may help you)


### Installation
```
git clone https://github.com/eladrich/pixel2style2pixel.git
```
cd pixel2style2pixel


## Datasets
1. ADE20K
    * paired images and segmentation maps
    * 3千多種class & 25k 張各式場景圖片
    * 只挑選較多室外場景的類別使用
        * nature_landscape
        * urban
        * industrial
    * https://groups.csail.mit.edu/vision/datasets/ADE20K/
2. OST300
    * 只挑選較多室外場景的類別使用
        * water
        * sky
        * mountain
        * lakesea
        * cloud
        * plant
        * lawn
        * grass
    * 約6000張場景圖
    * http://mmlab.ie.cuhk.edu.hk/projects/SFTGAN/
3. LHQ Dataset
    * 約9萬多張高清大自然場景圖
    * https://universome.github.io/alis
4. landscape picture
    * 約4500張大自然場景圖
    * https://www.kaggle.com/datasets/arnaud58/landscape-pictures

### Models

#### Image Generator
1. SPADE
    * 根據Paper自己實做整個網路
    * https://github.com/NVlabs/SPADE
2. psp encoder + StyleGANv2
    1. psp encoder
        * 根據paper實做Encoder
        * 其餘處理與Decoder使用官方程式碼，使用新的資料集進行訓練
        * 調整程式碼以融入系統中
        * https://github.com/eladrich/pixel2style2pixel
    2. StyleGANv2
        * 使用github專案程式碼，使用新的資料集進行訓練
        * 調整程式碼以融入系統中
        * https://github.com/rosinality/stylegan2-pytorch



### Docker
#### Goal
讓User可以直接下 ```docker-compose up``` 的指令一鍵部署

#### Frontend
* 預計基於node:14與nginx:alpine
    * https://hub.docker.com/_/node
    * https://hub.docker.com/_/nginx
* 發布(目前仍在更新中) : 
    * https://hub.docker.com/r/popopopony/ccclub2022fall_frontend

#### Backend
* 預計基於pytorch/pytorch:latest，加上fastapi需要的套件與指令
    * https://hub.docker.com/r/pytorch/pytorch
* 發布(目前仍在更新中) : 
    * https://hub.docker.com/r/popopopony/ccclub2022fall_backend






## Project Timeline
![](https://i.imgur.com/H5WhiAz.png)

## References
* Paper
    * Semantic Image Synthesis with Spatially-Adaptive Normalization
    * Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation
    * A Style-Based Generator Architecture for Generative Adversarial Networks
    * Analyzing and Improving the Image Quality of StyleGAN
    * Real-time Semantic Segmentation Network Inspired from PID Controller

* Others
    * https://groups.csail.mit.edu/vision/datasets/ADE20K/
    * http://mmlab.ie.cuhk.edu.hk/projects/SFTGAN/
    * https://universome.github.io/alis
    * https://www.kaggle.com/datasets/arnaud58/landscape-pictures
    * https://github.com/NVlabs/SPADE
    * https://github.com/eladrich/pixel2style2pixel
    * https://github.com/rosinality/stylegan2-pytorch
    * https://hub.docker.com/_/node
    * https://hub.docker.com/_/nginx
    * https://hub.docker.com/r/pytorch/pytorch