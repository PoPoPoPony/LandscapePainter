###### tags: `ccClub2022fall`

# Landscape Painter(github README.md)


> We provide a web application to transfer the image from a segmentation map to a photorealistic image. In the application, you can choose the  SPADE or the PsP encoder with StyleGANv2 to generate your painting. We also provide the docker images and the checkpoint files that can quickly build the application on your own. 

![](https://i.imgur.com/ADUYIPA.png)
<div style='text-align: center'>By default, we only provide 5 classes to draw. You can reference the <a href="https://groups.csail.mit.edu/vision/datasets/ADE20K/">ADE20K</a> document if you want to change the class.<p><b>(Note that the drawing color is equal to the label color in ADE20K)</b></div>

## Table of Contents
- [Landscape Painter(github README.md)](#landscape-paintergithub-readmemd)
  - [Table of Contents](#table-of-contents)
  - [Goals](#goals)
  - [Architecture](#architecture)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Nvidia GPU driver Installation](#nvidia-gpu-driver-installation)
    - [WSL2 Installation](#wsl2-installation)
  - [Repository Structure](#repository-structure)
  - [Project Timeline](#project-timeline)
  - [References](#references)


## Goals
1. Develop a web application
2. Build a demo page
3. Transfer the image from a segmentation map to a photorealistic image
4. Provide docker images and well-trained checkpoint files 

## Architecture
![](https://i.imgur.com/zvaTAqU.png)


## Getting Started
### Prerequisites
* Computer
* Nvidia GPU, driver need to be installed. (The tutorial in [here](#WSL2-Installation) may help you)
* For Windows user, WSL2 need to be installed. (The tutorial in [here](#WSL2-Installation) may help you)


### Installation

1. Clone the project

```
git clone https://github.com/PoPoPoPony/LandscapePainter.git
```
* ***Note that git should be installed***
* If you didn't install the git, please refer to https://git-scm.com/



2. Download the checkpoint files, check <a href="https://drive.google.com/file/d/1jcnEqMO_6UWjgC-EUvHPqHa_vgamvWoL/view?usp=share_link">here</a>

3. unzip the ```ckpts.zip```in the ```LandscapePainter```, which will looks like [Repository structure](#Repository-Structure)

4. run the docker compose commend
```
cd LandscapePainter
```
```
docker-compose up
```
If you see this, you can start painting!
    ![](https://i.imgur.com/W9D1MI6.png)
* ***Note that downloading the envirovment(almost 18G) and initializing the models(backend) take time. Go to have a cup of coffee first!***

5. Painting!

### Nvidia GPU driver Installation
1. check your status, open the CMD and enter : 
```
nvidia-smi
```

If the info shows, 
![](https://i.imgur.com/XCLdfj4.png)




### WSL2 Installation






## Repository Structure
| Path     | Description    |
| ------------------------------------------------- |:------------------------------------------------------------------------- |
| PsP    | Modules for training PsP encoder      |
| SPADE       | SPADE    |
| backend  | Backend based on fastapi  |
| frontend   | Front end based on Vue and element-plus3     |
| &boxv;&nbsp;&boxv;&nbsp;&boxvr;&nbsp; Painter.vue <img width=200> | Setting the painter, if you want the change the class, please modify here |
| posters   | Images in github readme     |
| stylegan2-pytorch   | Modules for training StyleGANv2     |
| webcrawler_sophie   | Web crawler scripts provide by sophie     |
| webcrawler_wen   | Web crawler scripts provide by wen     |
| docker-compose.yml <img width=200>  | docker compose files     |
| PsP_ckpt  | PsP checkpoint file     |
| &boxvr;&nbsp; iteration_200000.pt <img width=300> | The checkpoint files for PsP encoder. You can set your own pt file after training your PsP encoder |
| SPADE_ckpt  | SPADE checkpoint file     |
| &boxvr;&nbsp; epoche007 .pt <img width=300> | The checkpoint files for SPADE. You can set your own pt file after training your SPADE |





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