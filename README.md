# NexToU: Efficient Topology-Aware U-Net for Medical Image Segmentation
[![Paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2305.15911)

## :bulb: News
* **(September 19, 2023):** Launched NexToU architecture and training codes for [nnU-Net V2](https://github.com/PengchengShi1220/NexToU/tree/NexToU_nnunetv2).
* **(June 14, 2023):** Updated NexToU installation and running demo.
* **(May 26, 2023):** Released NexToU architecture and training codes for [nnU-Net V1](https://github.com/PengchengShi1220/NexToU/tree/NexToU_nnunetv1).

<p align="center">
  <img src="assets/NexToU.png" alt="NexToU" width="180"/>
</p>

> **Abstract:** *Convolutional neural networks (CNN) and Transformer variants have emerged as the leading medical image segmentation backbones. Nonetheless, due to their limitations in either preserving global image context or efficiently processing irregular shapes in visual objects, these backbones struggle to effectively integrate information from diverse anatomical regions and reduce inter-individual variability, particularly for the vasculature. Motivated by the successful breakthroughs of graph neural networks (GNN) in capturing topological properties and non-Euclidean relationships across various fields, we propose NexToU, a novel hybrid architecture for medical image segmentation. NexToU comprises improved Pool GNN and Swin GNN modules from Vision GNN (ViG) for learning both global and local topological representations while minimizing computational costs. To address the containment and exclusion relationships among various anatomical structures, we reformulate the topological interaction (TI) module based on the nature of binary trees, rapidly encoding the topological constraints into NexToU. Extensive experiments conducted on three datasets (including distinct imaging dimensions, disease types, and imaging modalities) demonstrate that our method consistently outperforms other state-of-the-art (SOTA) architectures.* 

## NexToU Architecture Overview

The proposed NexToU architecture follows a hierarchical U-shaped encoder-decoder structure that includes purely convolutional modules and x topological ones. NexToU incorporates improved Pool GNN and Swin GNN modules from [Vision GNN (ViG)](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch), designed to learn both global and local topological representations while minimizing computational costs. It reformulates the [topological interaction (TI)](https://github.com/TopoXLab/TopoInteraction) module based on the nature of binary trees, rapidly encoding the topological constraints into NexToU. This unique approach enables effective handling of containment and exclusion relationships among various anatomical structures. To maintain consistency in data augmentation and post-processing, we base our NexToU architecture on the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) framework which can automatically configure itself for any new medical image segmentation task.
![NexToU Architecture](/assets/NexToU_architecture.jpg)

## Results

NexToU outperforms state-of-the-art (SOTA) methods in medical image segmentation across various tasks. On the Beyond the Cranial Vault (BTCV) dataset, NexToU achieved an average Dice Similarity Coefficient (DSC) of 87.84% and an average Hausdorff Distance (HD) of 6.33 mm. Additionally, it reached the highest DSC of 74.19% on the intracranial arteries (ICA) dataset. Best results are in bold.
![BTCV_results](/assets/BTCV_results.jpg)
![ICA_results](/assets/ICA_results.jpg)

## Qualitative Visualizations

NexToU outperforms alternative models in inter-class boundary segmentation and parameter efficiency. This architecture offers superior segmentation results with fewer misclassifications, specifically in organs like the spleen, liver, stomach, and middle cerebral artery.
![qualitative_visualizations](/assets/qualitative_visualizations.jpg)

## Usage

NexToU consists of several main components. The following links will take you directly to the core parts of the codebase:

- Network Architecture: The network architecture can be found in [NexToU.py](https://github.com/PengchengShi1220/NexToU/blob/NexToU_nnunetv2/network_architecture/NexToU.py) and [NexToU_Encoder_Decoder.py](https://github.com/PengchengShi1220/NexToU/blob/NexToU_nnunetv2/network_architecture/NexToU_Encoder_Decoder.py).
- Network Training: The file responsible for network training is [nnUNetTrainer_nextou.py](https://github.com/PengchengShi1220/NexToU/blob/NexToU_nnunetv2/nnUNetTrainer/nnUNetTrainer_nextou.py).
- Binary Topological Interaction (BTI) Loss Function: The BTI loss function is in [bti_loss.py](https://github.com/PengchengShi1220/NexToU/blob/NexToU_nnunetv2/loss/bti_loss.py).

To incorporate the functionalities of NexToU with nnUNet, follow the steps given below:

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

If you're using the `3d_fullres_nextou` configuration, make sure to update your `nnUNet_preprocessed/DatasetXX/nnUNetPlans.json` file. The channel count should be a multiple of 3. You can add the following JSON snippet to your existing `nnUNetPlans.json`. Have a look at the example provided in the following link: [nnUNetPlans.json](https://github.com/PengchengShi1220/NexToU/blob/0e9f558c97044917a2353d81133dd39e1b003c8d/nnUNetPlans.json#L426):

```json
"3d_fullres_nextou": {
    "inherits_from": "3d_fullres",
    "patch_size": [
        64,
        224,
        192
    ],
    "UNet_base_num_features": 24,
    "unet_max_num_features": 312
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

