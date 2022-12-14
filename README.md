# Landscape Painter

## Goals

1. 部署一個網頁應用
2. 讓使用者可以透過色塊輕鬆繪製真實場景
3. 建立易重建的環境(based on docker)
4. 建立完整的說明(in github readme)

## User Flows

![https://i.imgur.com/Ix5D9KE.png](https://i.imgur.com/Ix5D9KE.png)

## Architecture

![https://i.imgur.com/1ChyYGq.png](https://i.imgur.com/1ChyYGq.png)

## Work Description

### Datasets

1. ADE20K
    - paired image and segmentation map
    - 3千多種class & 25k 張各式場景圖片
    - 只挑選較多室外場景的類別使用
        - nature_landscape
        - urban
        - industrial
    - [https://groups.csail.mit.edu/vision/datasets/ADE20K/](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
2. OST300
    - 只挑選較多室外場景的類別使用
        - water
        - sky
        - mountain
        - lakesea
        - cloud
        - plant
        - lawn
        - grass
    - 約6000張場景圖
    - [http://mmlab.ie.cuhk.edu.hk/projects/SFTGAN/](http://mmlab.ie.cuhk.edu.hk/projects/SFTGAN/)
3. LHQ Dataset
    - 約9萬多張高清大自然場景圖
    - [https://universome.github.io/alis](https://universome.github.io/alis)
4. landscape picture
    - 約4500張大自然場景圖
    - [https://www.kaggle.com/datasets/arnaud58/landscape-pictures](https://www.kaggle.com/datasets/arnaud58/landscape-pictures)
5. COCO 2017
    - ***因ADE20K的類別太多，每個類別出現的頻率不夠高，導致PsP encoder無法正確操控特徵，因此另外使用COCO再train一個模型***
    - 約20萬張圖片 & 80個class
    - [https://cocodataset.org/#home](https://cocodataset.org/#home)
    - [https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)

### Models

### Segmentation model

1. PIDNet
    - 直接使用PIDNet的官方實做版本，調整部分程式以融入系統中
    - [https://github.com/XuJiacong/PIDNet](https://github.com/XuJiacong/PIDNet)
2. deepLabv3
    - 直接使用Pytorch內建的deeplabv3(base on ResNet50 backbone)，調整部分程式以融入系統中
    - [https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/)

### Image Generator

1. SPADE
    - 根據Paper自己實做整個網路
    - [https://github.com/NVlabs/SPADE](https://github.com/NVlabs/SPADE)
2. psp encoder + StyleGANv2
    1. psp encoder
        - 根據paper實做Encoder
        - 其餘處理與Decoder使用官方程式碼，使用新的資料集進行訓練
        - 調整程式碼以融入系統中
        - [https://github.com/eladrich/pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel)
    2. StyleGANv2
        - 使用github專案程式碼，使用新的資料集進行訓練
        - 調整程式碼以融入系統中
        - [https://github.com/rosinality/stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch)

### Web

### Frontend

1. Vue3
    - [https://vuejs.org/](https://vuejs.org/)
2. Element Plus 3
    - 好用的CSS框架，提供許多方便的元件
    - [https://element-plus.org/en-US/](https://element-plus.org/en-US/)

### Backend

1. FastAPI
    - 自動產生Swagger UI真的太神拉，測試起來超方便XD
    - 
        
        ![https://i.imgur.com/ff0yQFV.png](https://i.imgur.com/ff0yQFV.png)
        
    - [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)

### Docker

### Goal

讓User可以直接下 `docker-compose up` 的指令一鍵部署

### Frontend

- 預計基於node:14與nginx:alpine
    - [https://hub.docker.com/_/node](https://hub.docker.com/_/node)
    - [https://hub.docker.com/_/nginx](https://hub.docker.com/_/nginx)
- 發布(目前仍在更新中) :
    - [https://hub.docker.com/r/popopopony/ccclub2022fall_frontend](https://hub.docker.com/r/popopopony/ccclub2022fall_frontend)

### Backend

- 預計基於pytorch/pytorch:latest，加上fastapi需要的套件與指令
    - [https://hub.docker.com/r/pytorch/pytorch](https://hub.docker.com/r/pytorch/pytorch)
- 發布(目前仍在更新中) :
    - [https://hub.docker.com/r/popopopony/ccclub2022fall_backend](https://hub.docker.com/r/popopopony/ccclub2022fall_backend)

## Project Timeline

![https://i.imgur.com/H5WhiAz.png](https://i.imgur.com/H5WhiAz.png)

## References

- Paper
    - Semantic Image Synthesis with Spatially-Adaptive Normalization
    - Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation
    - A Style-Based Generator Architecture for Generative Adversarial Networks
    - Analyzing and Improving the Image Quality of StyleGAN
    - Real-time Semantic Segmentation Network Inspired from PID Controller
- Others
    - [https://groups.csail.mit.edu/vision/datasets/ADE20K/](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
    - [http://mmlab.ie.cuhk.edu.hk/projects/SFTGAN/](http://mmlab.ie.cuhk.edu.hk/projects/SFTGAN/)
    - [https://universome.github.io/alis](https://universome.github.io/alis)
    - [https://www.kaggle.com/datasets/arnaud58/landscape-pictures](https://www.kaggle.com/datasets/arnaud58/landscape-pictures)
    - [https://github.com/NVlabs/SPADE](https://github.com/NVlabs/SPADE)
    - [https://github.com/eladrich/pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel)
    - [https://github.com/rosinality/stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch)
    - [https://hub.docker.com/_/node](https://hub.docker.com/_/node)
    - [https://hub.docker.com/_/nginx](https://hub.docker.com/_/nginx)
    - [https://hub.docker.com/r/pytorch/pytorch](https://hub.docker.com/r/pytorch/pytorch)