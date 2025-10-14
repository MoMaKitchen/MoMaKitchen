<div align="center">

<h1>🎇MoMa-Kitchen: A 100K+ Benchmark for Affordance-Grounded Last-Mile Navigation in Mobile Manipulation</h1>


<br>

<div>
    <a href='https://arxiv.org/abs/2503.11081' target='_blank'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</div>

</div>


## Project Overview
We present MoMa-Kitchen, a benchmark dataset with over 100k auto-generated samples featuring affordance-grounded manipulation positions and egocentric RGB-D data, and propose NavAff, a lightweight model that learns optimal navigation termination for seamless manipulation transitions. Our approach generalizes across diverse robotic platforms and arm configurations, addressing the critical gap between navigation proximity and manipulation readiness in mobile manipulation. 

## <a name="todo"></a> Status

- [x] Paper uploaded to arXiv  
- [x] MoMa-Kitchen Dataset release
- [x] NavAff Model Training Code release
- [ ] Data Collection Code and Assets release

## Installation

### Conda Environment
1. Create a conda environment with Python 3.8
```bash
conda create --name MoMaKitchen python=3.8
conda activate MoMaKitchen
```
2. Install Pytorch 

Please change to your CUDA version
```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```
3. Install Main Dependencies
```bash
pip install -r requirements.txt
```

## Data Preparation

Download the robot info data from the Google Drive:
[24.7MB](https://drive.google.com/file/d/1YnfyRBSM9gOk8rmSHN8q40nHlfhNTUgY/view?usp=sharing)

Then unzip this file and change the `info_root` path in `config.yaml`.


Download the RGBD and processed point cloud data at this URL:
[https://huggingface.co/datasets/IPEC-COMMUNITY/MoMa-Kitchen-Data](https://huggingface.co/datasets/IPEC-COMMUNITY/MoMa-Kitchen-Data)

(Option) After downloading, you can delete the .git folder in the dataset directory to save space.
```bash
rm -rf .git 
```
Then remember to change the `rgbd_root` to Path/To/Your/Datafolder in `config.yaml`.


## Code
Start training 

```bash
bash train_on_ali.sh
```

The training process costs nearly 12h on a single A100 GPU.
