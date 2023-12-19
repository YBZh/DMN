## few-shot results, training required, res50 backbone.
# PLOT: PROMPT LEARNING WITH OPTIMAL TRANS- PORT FOR VISION-LANGUAGE MODELS
# TIP, CLIP-Adapter, APE, Coop,


######## results of tip, tip-f, and clip-adapter can be achieved from   https://github.com/gaopengcuhk/Tip-Adapter/blob/main/exp.log

import matplotlib.pyplot as plt
import numpy as np
import torch
import ipdb
def get_datasets_mean(all_list):
    # ipdb.set_trace()
    to_tensor = torch.Tensor(all_list)
    mean_tensor = to_tensor.mean(0)
    return mean_tensor.tolist()


datasets = ['ImageNet', 'Flowers102', 'DTD', 'OxfordPets', 'StanfordCars', 'UCF101', 'Caltech101', 'Food101', 'SUN397', 'FGVCAircraft', 'EuroSAT', 'Mean']
methods = ['Tip-X', 'Tip-Adapter', 'APE', 'DSMN']
excel_path = '/Users/zhangyabin/Nutstore Files/我的坚果云/paper draft/CVPR2024/training_free_res50.xlsx'
results_path = '/Users/zhangyabin/Nutstore Files/我的坚果云/paper draft/CVPR2024/CLIP_tuning/images/results_path_resnet_training_free/'
# results_path = '/Users/zhangyabin/Nutstore Files/我的坚果云/paper draft/CVPR2024/'
import pandas as pd
import numpy as np

# Load the Excel file into a DataFrame
df = pd.read_excel(excel_path, header=None)

# Function to find the index of the next empty row after a given start index
def find_next_empty_row(start_index):
    empty_rows = df[df.isnull().all(axis=1)].index
    for idx in empty_rows:
        if idx > start_index:
            return idx
    return None

all_cate = {}
for method in methods:
    print(method)
    # ipdb.set_trace()
    # Find rows containing "APE-F" and "CoOp" labels
    method_start = df[df[0] == method].index[0]
    method_end = find_next_empty_row(method_start)

    # Extract results data
    method_results = df.iloc[method_start+1: method_end, 1:13].values.tolist()
    method_results = np.array(method_results).T.tolist()  ## [11 tasks+1mean * 5 shot]
    all_cate[method] = method_results

shots = [0, 1, 2, 4, 8, 16]
shots_calip_zero_shot = [0]  # Only for CALIP and Zero-shot CLIP
### copyed from: https://github.com/gaopengcuhk/Tip-Adapter/blob/main/exp.log


zero_shot_clip_results = [60.32, 66.10, 40.07, 85.83, 55.71, 61.33, 83.94, 77.32, 58.53, 17.10, 37.54, 58.53]
# zero_shot_clip_mean = []
for i in range(len(datasets)):
    dataset_name = datasets[i]
    our_dsmn = all_cate['DSMN'][i]
    ape = all_cate['APE'][i]
    tip = all_cate['Tip-Adapter'][i]
    tip_x = all_cate['Tip-X'][i]
    zero_shot_clip = [zero_shot_clip_results[i]]
    # Plotting
    plt.figure(figsize=(4,3))
    plt.plot(shots[1:], our_dsmn[1:], '*--', label='DMN-TF (Ours)', color='red')
    # plt.plot(shots, our_dsmn, '*--', label='DSMN (Ours)', color='red')
    plt.plot(shots[1:], ape, 'v--', label='APE', color='cyan')
    plt.plot(shots[1:], tip, 'p--', label='TIP-Adapter', color='magenta')
    # plt.plot(shots[1:], coop, 'o-', label='CoOp', color='blue')
    plt.plot(shots[1:], tip_x, 's--', label='TIP-X', color='black')
    # Plotting only points for CALIP and Zero-shot CLIP
    # plt.scatter(shots_calip_zero_shot, calip, marker='v', label='CALIP', color='purple')
    plt.scatter(shots_calip_zero_shot, zero_shot_clip, marker='D', label='Zero-shot CLIP', color='brown')
    # Set labels, title, legend, etc.
    plt.xlabel('Shots Number')
    plt.ylabel('Accuracy (%)')
    if dataset_name == 'Mean':
        plt.title('Average Results of 11 Datasets')
    else:
        plt.title('Training-free Results on ' + dataset_name)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks([0, 1, 2, 4, 8, 16])
    # max_num = max(our_dsmn_t[-1], ape_t[-1], tip_f[-1])
    # min_num = min(tip_f[0], coop[0], clip_adapter[0], coop[1])
    # plt.ylim(min_num - (max_num-min_num)*0.2)
    plt.tight_layout()
    # Show the plot
    # plt.show()
    plt.savefig(results_path+'resnet_tf_' + dataset_name + '.pdf', dpi=300, bbox_inches='tight', transparent=True)

# our_dsmn_t= [      74.20, 76.26, 78.62, 80.98, 83.10]  ## to refine.
# # our_dsmn=   [70.5, 72.67, 74.51, 76.97, 78.88, 80.22]  ## to refine.
# ########### 1/2/4/8/16 shot.
# coop =           [60.13, 62.71, 67.25, 70.62, 73.79]
# clip_adapter =   [69.27, 72.58, 75.37, 78.89, 81.79]
# tip_f =          [72.32, 75.29, 78.35, 80.69, 82.87]
