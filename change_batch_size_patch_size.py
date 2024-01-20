import pickle
import argparse
import numpy as np
import os
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name

# python change_batch_size_patch_size.py 3d_fullres Task111_Synapse_CT -bs 2 -ps 48 192 192

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("network")
    parser.add_argument("task", help="can be task name or task id")
    parser.add_argument("-bs", help="batch_size", default=None)
    parser.add_argument("-ps", nargs='+', help="patch_size", default=None)
    
    args = parser.parse_args()

    network = args.network
    task = args.task
    new_batch_size = args.bs
    new_patch_size = args.ps

    if isinstance(new_patch_size, list):
        if new_patch_size[0] == 'all' and len(new_patch_size) == 1:
            pass
        else:
            new_patch_size = [int(i) for i in new_patch_size]
    else:
        new_patch_size = None

    nnUNet_preprocessed_path = os.environ["nnUNet_preprocessed"]

    if network == "2d":
        network_name = "2D"
    else:
        network_name = "3D"

    if not task.startswith("Task"):
        task_id = int(task)
        task = convert_id_to_task_name(task_id)

    pkl_path = nnUNet_preprocessed_path + "/" + task + "/" + "nnUNetPlansv2.1_plans_" + network_name + ".pkl"
    f = open(pkl_path, 'rb')
    plans = pickle.load(f)

    print("before change:")
    print("stage 0, batch_size: ", plans['plans_per_stage'][0]['batch_size'])
    print("stage 0, patch_size: ", plans['plans_per_stage'][0]['patch_size'])
    try:
        print("stage 1, batch_size: ", plans['plans_per_stage'][1]['batch_size'])
        print("stage 1, patch_size: ", plans['plans_per_stage'][1]['patch_size'])
    except:
        pass

    plans = load_pickle(pkl_path)

    if new_batch_size != None:
        plans['plans_per_stage'][0]['batch_size'] = int(new_batch_size)
        try:
            plans['plans_per_stage'][1]['batch_size'] = int(new_batch_size)
        except:
            pass
    else:
        pass
    
    if new_patch_size != None:
        if network == "2d":
            plans['plans_per_stage'][0]['patch_size'] = np.array((new_patch_size[0], new_patch_size[1]))  
            try:
                plans['plans_per_stage'][1]['patch_size'] = np.array((new_patch_size[0], new_patch_size[1])) 
            except:
                pass
        else:
            plans['plans_per_stage'][0]['patch_size'] = np.array((new_patch_size[0], new_patch_size[1], new_patch_size[2]))  
            try:
                plans['plans_per_stage'][1]['patch_size'] = np.array((new_patch_size[0], new_patch_size[1], new_patch_size[2])) 
            except:
                pass
    else:
        pass

    print("after change:")
    print("stage 0, batch_size: ", plans['plans_per_stage'][0]['batch_size'])
    print("stage 0, patch_size: ", plans['plans_per_stage'][0]['patch_size'])
    try:
        print("stage 1, batch_size: ", plans['plans_per_stage'][1]['batch_size'])
        print("stage 1, patch_size: ", plans['plans_per_stage'][1]['patch_size'])
    except:
        pass

    save_pickle(plans, join(nnUNet_preprocessed_path + "/" + task + "/",
                            "nnUNetPlansv2.1_plans_" + network_name + ".pkl"))

if __name__ == "__main__":
    main()
