# NexToU: Efficient Topology-Aware U-Net for Medical Image Segmentation
<p align="center">
  <img src="assets/NexToU.png" alt="NexToU" width="200"/>
</p>
NexToU is a novel hybrid architecture for medical image segmentation that combines the strengths of Convolutional Neural Networks (CNN), and Graph Neural Networks (GNN) variants. It addresses the limitations in preserving global image context and efficiently processing irregular shapes, a common struggle in the field of medical image analysis.

## Motivation 

Despite the advancements in CNN and Transformer variants, they often struggle to effectively integrate information from diverse anatomical regions and reduce inter-individual variability, particularly for vasculature. Recognizing the capabilities of GNN in capturing topological properties and non-Euclidean relationships across various fields, we developed NexToU to overcome these challenges.

## About NexToU

NexToU incorporates improved Pool GNN and Swin GNN modules from [Vision GNN (ViG)](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch), designed to learn both global and local topological representations while minimizing computational costs. It reformulates the [topological interaction (TI)](https://github.com/TopoXLab/TopoInteraction) module based on the nature of binary trees, rapidly encoding the topological constraints into NexToU. This unique approach enables effective handling of containment and exclusion relationships among various anatomical structures. To maintain consistency in data augmentation and post-processing, we base our NexToU architecture on the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) framework which can automatically configure itself for any new medical image segmentation task.

## Performance 

NexToU has been extensively tested on three different datasets, each featuring distinct imaging dimensions, disease types, and imaging modalities. In all cases, NexToU outperformed other state-of-the-art (SOTA) architectures in medical image segmentation tasks.

## Usage

NexToU consists of several main components. The following links will take you directly to the core parts of the codebase:

- Network Architecture: The network architecture can be found in [NexToU.py](https://github.com/PengchengShi1220/NexToU/blob/main/network_architecture/NexToU.py).
- Network Training: The file responsible for network training is [nnUNetTrainerV2_nextou.py](https://github.com/PengchengShi1220/NexToU/blob/main/network_training/nnUNetTrainerV2_nextou.py).
- Binary Topological Interaction (BTI) Loss Function: The BTI loss function is in [BTI_loss.py](https://github.com/PengchengShi1220/NexToU/blob/main/loss_functions/BTI_loss.py).

To use NexToU in your own projects, you may import the relevant components from these files as per your requirements. Please make sure to comply with the license agreement while using the code.

If you have any issues or questions, feel free to open an issue on our GitHub repository.

## License

NexToU is licensed under the Apache License 2.0. For more information, please see the [LICENSE](LICENSE) file in this repository.

## References

- NexToU: Efficient Topology-Aware U-Net for Medical Image Segmentation. Available at: [https://arxiv.org/abs/2305.15911](https://arxiv.org/abs/2305.15911)
