# 2022.03.16-Changed for training NexToU with BTI loss on BTCV datasets
#            Harbin Institute of Technology (Shenzhen), <pcshi@stu.hit.edu.cn>

#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import numpy as np
from torch import nn
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.loss_functions.BTI_loss import DC_and_CE_and_BTI_Loss
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.network_training.nnUNetTrainerV2_nextou import nnUNetTrainerV2_NexToU


class nnUNetTrainerV2_NexToU_BTI_Synapse(nnUNetTrainerV2_NexToU):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 1000
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.pin_memory = True

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights

            # ECCV 2022 Oral: Learning Topological Interactions for Multi-Class Medical Image Segmentation
            # λti = 1e-4 in the 2D setting, and λti = 1e-6 in the 3D settings.
            if self.threeD:
                dim = 3
                connectivity = 26
                lambda_ti = 1e-6
            else:
                dim = 2
                connectivity = 8
                lambda_ti = 1e-4

            # Synapse(BTCV datasets):
            inclusion_list = []
            exclusion_list = [[[1, 3, 5, 7, 8, 11, 13], [2, 4, 6, 9, 10, 12]], [[1, 3, 11, 13], [5, 7, 8]], [[1, 3], [11, 13]], [1, 3], [11, 13], [[5, 8], [7]], [5, 8], [[4, 6, 10], [2, 9, 12]], [[4, 6], [10]], [4, 6], [[9, 12], [2]], [9, 12]]

            self.loss = DC_and_CE_and_BTI_Loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {}, 
                                              {'dim': dim, 'connectivity': connectivity, 'inclusion': inclusion_list, 'exclusion': exclusion_list, 'min_thick': 1}, 
                                            weight_ce=1, weight_dice=1, weight_ti=lambda_ti)
            
            self.print_to_log_file("dim: %s" % str(dim))
            self.print_to_log_file("connectivity: %s" % str(connectivity))
            self.print_to_log_file("lambda_ti: %s" % str(lambda_ti))
            self.print_to_log_file("inclusion_list: %s" % str(inclusion_list))
            self.print_to_log_file("exclusion_list_len: %s" % str(len(exclusion_list)))
            self.print_to_log_file("exclusion_list: %s" % str(exclusion_list))

            # now wrap the loss
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True
