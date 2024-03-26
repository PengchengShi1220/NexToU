from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_NexToU import nnUNetTrainer_NexToU


class nnUNetTrainer_NexToU_NoMirroring(nnUNetTrainer_NexToU):
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes
