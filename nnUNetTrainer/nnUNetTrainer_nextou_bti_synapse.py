import numpy as np
import torch
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.compound_bti_loss import DC_and_CE_and_BTI_Loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_nextou import nnUNetTrainer_NexToU

class nnUNetTrainer_NexToU_BTI_Synapse(nnUNetTrainer_NexToU):
    def make_tensors(self, lists, device):
        if not lists:
            return lists
        elif isinstance(lists[0], list):
            return [self.make_tensors(sublist, device) for sublist in lists]
        else:
            return torch.tensor(lists).to(device)
            
    def _build_loss(self):

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()

        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        # ECCV 2022 Oral: Learning Topological Interactions for Multi-Class Medical Image Segmentation
        # λti = 1e-4 in the 2D setting, and λti = 1e-6 in the 3D settings.
        if dim == 3:
            connectivity = 26
            lambda_ti = 1e-6
        else:
            connectivity = 8
            lambda_ti = 1e-4

        # Synapse(BTCV datasets):
        inclusion_list = []
        exclusion_list = [[[1, 3, 5, 7, 8, 11, 13], [2, 4, 6, 9, 10, 12]], [[1, 3, 11, 13], [5, 7, 8]], [[1, 3], [11, 13]], [
                    1, 3], [11, 13], [[5, 8], [7]], [5, 8], [[4, 6, 10], [2, 9, 12]], [[4, 6], [10]], [4, 6], [[9, 12], [2]], [9, 12]]
        
        inclusion_list = self.make_tensors(inclusion_list, self.device)
        exclusion_list = self.make_tensors(exclusion_list, self.device)

        loss = DC_and_CE_and_BTI_Loss({'batch_dice': self.configuration_manager.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {},
                                    {'dim': dim, 'connectivity': connectivity, 'inclusion': inclusion_list, 'exclusion': exclusion_list, 'min_thick': 1},
                                    weight_ce=1, weight_dice=1, weight_ti=lambda_ti, ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        self.print_to_log_file("dim: %s" % str(dim))
        self.print_to_log_file("connectivity: %s" % str(connectivity))
        self.print_to_log_file("lambda_ti: %s" % str(lambda_ti))
        self.print_to_log_file("inclusion_list: %s" % str(inclusion_list))
        self.print_to_log_file("exclusion_list_len: %s" %
                                str(len(exclusion_list)))
        self.print_to_log_file("exclusion_list: %s" % str(exclusion_list))


        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss


           

