import pickle
import argparse
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name

# python change_split_file.py Task111_Synapse_CT
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="can be task name or task id")
    
    args = parser.parse_args()
    task = args.task

    nnUNet_preprocessed_path = os.environ["nnUNet_preprocessed"]

    if not task.startswith("Task"):
        task_id = int(task)
        task = convert_id_to_task_name(task_id)
    
    if task == 'Task111_Synapse_CT':
        train_case_name_list = ['case0006', 'case0007', 'case0009', 'case0010', 'case0021', 'case0023', 'case0024', 'case0026', 'case0027', 'case0031', 'case0033', 'case0034', 'case0039', 'case0040', 'case0005', 'case0028', 'case0030', 'case0037']
        val_case_name_list = ['case0001', 'case0002', 'case0003', 'case0004', 'case0008', 'case0022', 'case0025', 'case0029', 'case0032', 'case0035', 'case0036', 'case0038']
    else:
        pass

    # print("train_case_name_list: ", train_case_name_list)
    # print("val_case_name_list: ", val_case_name_list)
    print("len(train_case_name_list): ", len(train_case_name_list))
    print("len(val_case_name_list): ", len(val_case_name_list))
    pkl_path = nnUNet_preprocessed_path + "/" + task + "/" + "splits_final.pkl"
    f = open(pkl_path, 'rb')
    b = pickle.load(f)
    b[0]['train'] = np.array(train_case_name_list)
    b[0]['val'] = np.array(val_case_name_list)

    save_pickle(b, join(nnUNet_preprocessed_path + "/" + task + "/",
                            "splits_final.pkl"))

if __name__ == "__main__":
    main()
