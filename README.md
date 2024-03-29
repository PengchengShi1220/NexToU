# NexToU: Efficient Topology-Aware U-Net for Medical Image Segmentation

<div align="center">
  
  <img src="assets/NexToU.png" width="180">
  
  | 📃 [**Paper**](https://arxiv.org/abs/2305.15911) | 📂 [**Weight Files**](https://drive.google.com/drive/folders/1VcH00e-4KgVUOJUJckZEChD0RWwLRyW6?usp=sharing) |
</div>

## :bulb: News
* **(Mar 26, 2024):** [NexToU v1.0](https://github.com/PengchengShi1220/NexToU/releases/tag/v1.0.0) release, based on [nnU-Net v2.0](https://github.com/MIC-DKFZ/nnUNet/releases/tag/v2.0).
* **(October 13, 2023):** :trophy: :tada: Our NexToU-based solution won the second place 🥈 in both the MICCAI 2023 [TopCoW 🐮](https://topcow23.grand-challenge.org/evaluation/finaltest-cta-multiclass/leaderboard/) and MICCAI 2023 [CROWN 👑](https://crown.isi.uu.nl/leaderboard/) Challenge.
* **(September 19, 2023):** Launched NexToU architecture and training codes for [nnU-Net V2](https://github.com/PengchengShi1220/NexToU/tree/NexToU_nnunetv2).
* **(June 14, 2023):** Updated NexToU installation and running demo.
* **(May 26, 2023):** Released NexToU architecture and training codes for [nnU-Net V1](https://github.com/PengchengShi1220/NexToU/tree/NexToU_nnunetv1).

## Overview

The proposed NexToU architecture follows a hierarchical U-shaped encoder-decoder structure that includes purely convolutional modules and x topological ones. NexToU incorporates improved Pool GNN and Swin GNN modules from [Vision GNN (ViG)](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch), designed to learn both global and local topological representations while minimizing computational costs. It reformulates the [topological interaction (TI)](https://github.com/TopoXLab/TopoInteraction) module based on the nature of binary trees, rapidly encoding the topological constraints into NexToU. This unique approach enables effective handling of containment and exclusion relationships among various anatomical structures. To maintain consistency in data augmentation and post-processing, we base our NexToU architecture on the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) framework which can automatically configure itself for any new medical image segmentation task.
![NexToU Architecture](/assets/NexToU_architecture.jpg)

## Usage

NexToU consists of several main components. The following links will take you directly to the core parts of the codebase:

- Network Architecture: The network architecture can be found in [NexToU.py](https://github.com/PengchengShi1220/NexToU/blob/NexToU_nnunetv2/network_architecture/NexToU.py) and [NexToU_Encoder_Decoder.py](https://github.com/PengchengShi1220/NexToU/blob/NexToU_nnunetv2/network_architecture/NexToU_Encoder_Decoder.py).
- Network Training: The file responsible for network training is [nnUNetTrainer_NexToU.py](https://github.com/PengchengShi1220/NexToU/blob/NexToU_nnunetv2/nnUNetTrainer/nnUNetTrainer_NexToU.py).
- Binary Topological Interaction (BTI) Loss Function: The BTI loss function is in [bti_loss.py](https://github.com/PengchengShi1220/NexToU/blob/NexToU_nnunetv2/loss/bti_loss.py). Specifically for the ICA dataset training, it is further adapted in [nnUNetTrainer_NexToU_BTI_ICA_NoMirroring.py](https://github.com/PengchengShi1220/NexToU/blob/NexToU_nnunetv2/nnUNetTrainer/nnUNetTrainer_NexToU_BTI_ICA_NoMirroring.py).

To integrate NexToU with nnUNet, you can directly download [NexToU v1.0](https://github.com/PengchengShi1220/NexToU/releases/tag/v1.0.0) (based on [nnU-Net v2.0](https://github.com/MIC-DKFZ/nnUNet/releases/tag/v2.0)) using:
```
wget https://github.com/PengchengShi1220/NexToU/releases/download/v1.0.0/NexToU_v1.0_nnU-Net_v2.0.tar.gz
```

Alternatively, follow these steps:

1. Clone the NexToU repository from GitHub using the command:
```
git clone https://github.com/PengchengShi1220/NexToU.git
```

2. Download v2.0 version of nnUNet using the command:
```
wget https://github.com/MIC-DKFZ/nnUNet/archive/refs/tags/v2.0.tar.gz
```

3. Extract the v2.0.tar.gz file using the command:
```
tar -zxvf v2.0.tar.gz
```

4. Copy the NexToU loss functions, network architecture, and network training code files to the corresponding directories in nnUNet-2.0 using the following commands:
```
cp NexToU-NexToU_nnunetv2/loss/* nnUNet-2.0/nnunetv2/training/loss/
cp NexToU-NexToU_nnunetv2/network_architecture/* nnUNet-2.0/nnunetv2/training/nnUNetTrainer/variants/network_architecture/
cp NexToU-NexToU_nnunetv2/nnUNetTrainer/* nnUNet-2.0/nnunetv2/training/nnUNetTrainer/
```

5. Install nnUNet-2.0 with the NexToU related function and run it:
```
cd nnUNet-2.0 && pip install -e .
```

If you're using the `3d_fullres_nextou` configuration, make sure to update your `nnUNet_preprocessed/DatasetXX/nnUNetPlans.json` file. The channel count should be a multiple of 3. You can add the following JSON snippet to your existing `nnUNetPlans.json`. Have a look at the example provided in the [nnUNetPlans.json](https://github.com/PengchengShi1220/NexToU/blob/75a5729f4d274887849346148ed0620c3f2c1cba/nnUNetPlans.json#L433):

```json
"3d_fullres_nextou": {
    "inherits_from": "3d_fullres",
    "patch_size": [
        64,
        224,
        192
    ],
    "UNet_base_num_features": 33,
    "unet_max_num_features": 324
}
```

For BTCV dataset:
```
nnUNetv2_train 111 3d_fullres_nextou 0 -tr nnUNetTrainer_NexToU_BTI_Synapse
```

For RAVIR dataset:
```
nnUNetv2_train 810 2d 0 -tr nnUNetTrainer_NexToU_BTI_RAVIR
```

For ICA dataset:
```
nnUNetv2_train 115 3d_fullres_nextou 0 -tr nnUNetTrainer_NexToU_BTI_ICA_noMirroring
```

You can use the relevant components of NexToU in your own projects by importing them from the respective files. Please ensure that you abide by the license agreement while using the code.

If you have any issues or questions, feel free to open an issue on our GitHub repository.


## License

NexToU is licensed under the Apache License 2.0. For more information, please see the [LICENSE](LICENSE) file in this repository.

## Citation
If you use NexToU in your research, please cite:

```
@article{shi2023nextou,
  title={NexToU: Efficient Topology-Aware U-Net for Medical Image Segmentation},
  author={Shi, Pengcheng and Guo, Xutao and Yang, Yanwu and Ye, Chenfei and Ma, Ting},
  journal={arXiv preprint arXiv:2305.15911},
  year={2023}
}
```

