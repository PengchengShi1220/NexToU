# NexToU Model Training with AbdomenAtlas1.0Mini Dataset

## Overview

This README provides instructions for setting up, training, and testing the NexToU model using the AbdomenAtlas1.0Mini dataset.

## Dataset

- **Dataset Name**: [AbdomenAtlas1.0Mini](https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas1.0Mini)

## Trained Model Weights and Code

The trained model weights (checkpoints) and relevant code are available for download:  
- **Google Drive Link**: [NexToU_Touchstone Benchmark](https://drive.google.com/drive/folders/1EPxLTso1fb1YSSnzuYkIUMu5NhuNhf0g)  

## Setup

1. **Software Requirements**:
   - Python 3.10
   - PyTorch 2.2.2+cu121
   - CUDA 12.1

2. **Hardware Requirements**:
   - NVIDIA GeForce RTX 3090 24G (or equivalent)

## Installation and Usage

1. **Install Dependencies**:
   ```bash
   cd SuPreM/benchmark_backbones/
   conda env create -f environment.yml
   conda activate suprem_NexToU_torch2.2_cu121
   ```

2. **Install nnUNet**:
   ```bash
   cd ..
   cd ..
   cd SuPreM/nnUNet
   pip install -e .
   ```

## Training on AbdomenAtlas1.0Mini

- **Dataset Used**: AbdomenAtlas1.0Mini
- **Training Configuration**:
  - Batch Size: 2
  - Patch Size: [160, 160, 96]
  - Epochs: 2000
  - Iterations per Epoch: 250
  - Training Duration: Approximately 7 days on an NVIDIA GeForce RTX 3090 24G.

## Data Preparation

- **Preprocessing Steps**:
  - Normalize images and convert to `.npy` and `.pkl` formats as per nnUNet.
  - 
- **Training Script**:
  ```bash
  cd SuPreM/benchmark_backbones/
  bash train.sh
  ```
  - Modify `datapath` and `preprocesspath` in the script to point to your dataset and preprocessing directories.

## Testing the Trained Model

- **Testing Script**:
  ```bash
  cd SuPreM/benchmark_backbones/
  bash test.sh
  ```
  - Modify `test_raw_folder` and `SuPreM_folder` in the script to point to your test data and output directories.

## Citations

- **NexToU**:
  ```bibtex
  @article{shi2023nextou,
    title={NexToU: Efficient Topology-Aware U-Net for Medical Image Segmentation},
    author={Shi, Pengcheng and Guo, Xutao and Yang, Yanwu and Ye, Chenfei and Ma, Ting},
    journal={arXiv preprint arXiv:2305.15911},
    year={2023}
  }
  ```

- **Touchstone Benchmark**:
  ```bibtex
  @misc{bassi2024touchstonebenchmarkrightway,
    title={Touchstone Benchmark: Are We on the Right Way for Evaluating AI Algorithms for Medical Segmentation?}, 
    author={Pedro R. A. S. Bassi and Wenxuan Li and Yucheng Tang and Fabian Isensee and Zifu Wang and Jieneng Chen and Yu-Cheng Chou and Yannick Kirchhoff and Maximilian Rokuss and Ziyan Huang and Jin Ye and Junjun He and Tassilo Wald and Constantin Ulrich and Michael Baumgartner and Saikat Roy and Klaus H. Maier-Hein and Paul Jaeger and Yiwen Ye and Yutong Xie and Jianpeng Zhang and Ziyang Chen and Yong Xia and Zhaohu Xing and Lei Zhu and Yousef Sadegheih and Afshin Bozorgpour and Pratibha Kumari and Reza Azad and Dorit Merhof and Pengcheng Shi and Ting Ma and Yuxin Du and Fan Bai and Tiejun Huang and Bo Zhao and Haonan Wang and Xiaomeng Li and Hanxue Gu and Haoyu Dong and Jichen Yang and Maciej A. Mazurowski and Saumya Gupta and Linshan Wu and Jiaxin Zhuang and Hao Chen and Holger Roth and Daguang Xu and Matthew B. Blaschko and Sergio Decherchi and Andrea Cavalli and Alan L. Yuille and Zongwei Zhou},
    year={2024},
    eprint={2411.03670},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2411.03670}, 
  }
  ```
