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
methods = ['CoOp', 'CLIP-Adapter', 'Tip-Adapter-F', 'APE-T', 'DSMN-T']
excel_path = '/Users/zhangyabin/Nutstore Files/我的坚果云/paper draft/CVPR2024/training_required_res50.xlsx'
results_path = '/Users/zhangyabin/Nutstore Files/我的坚果云/paper draft/CVPR2024/CLIP_tuning/images/results_path_resnet_training_required/'
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
    # Find rows containing "APE-F" and "CoOp" labels
    method_start = df[df[0] == method].index[0]
    method_end = find_next_empty_row(method_start)

    # Extract results data
    method_results = df.iloc[method_start+1: method_end, 1:13].values.tolist()
    method_results = np.array(method_results).T.tolist()  ## [11 tasks+1mean * 5 shot]
    all_cate[method] = method_results

print(all_cate.keys())
shots = [0, 1, 2, 4, 8, 16]
shots_calip_zero_shot = [0]  # Only for CALIP and Zero-shot CLIP
zero_shot_clip_results = [60.32, 66.02, 42.20, 85.83, 55.74, 61.35, 85.92, 77.32, 58.52, 17.10, 37.52]
for i in range(len(datasets)):
    dataset_name = datasets[i]
    our_dsmn_t = all_cate['DSMN-T'][i]
    ape_t = all_cate['APE-T'][i]
    tip_f = all_cate['Tip-Adapter-F'][i]
    coop = all_cate['CoOp'][i]
    clip_adapter = all_cate['CLIP-Adapter'][i]
    # zero_shot_clip = [zero_shot_clip_results[i]]
    # Plotting
    plt.figure(figsize=(4,3))
    plt.plot(shots[1:], our_dsmn_t, '*-', label='DMN (Ours)', color='red')
    # plt.plot(shots, our_dsmn, '*--', label='DSMN (Ours)', color='red')
    plt.plot(shots[1:], ape_t, 'v-', label='APE-T', color='cyan')
    plt.plot(shots[1:], tip_f, 'p-', label='TIP-Adapter-F', color='magenta')
    plt.plot(shots[1:], coop, 'o-', label='CoOp', color='blue')
    plt.plot(shots[1:], clip_adapter, 's-', label='CLIP-Adapter', color='black')
    # Plotting only points for CALIP and Zero-shot CLIP
    # plt.scatter(shots_calip_zero_shot, calip, marker='v', label='CALIP', color='purple')
    # plt.scatter(shots_calip_zero_shot, zero_shot_clip, marker='D', label='Zero-shot CLIP', color='brown')
    # Set labels, title, legend, etc.
    plt.xlabel('Shots Number')
    plt.ylabel('Accuracy (%)')
    if dataset_name == 'Mean':
        plt.title('Average Results of 11 Datasets')
    else:
        plt.title('Few-shot Results on ' + dataset_name)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks([0, 1, 2, 4, 8, 16])
    max_num = max(our_dsmn_t[-1], ape_t[-1], tip_f[-1])
    min_num = min(tip_f[0], coop[0], clip_adapter[0], coop[1], our_dsmn_t[0])
    plt.ylim(min_num - (max_num-min_num)*0.2)
    plt.tight_layout()
    # Show the plot
    # plt.show()
    plt.savefig(results_path+'resnet_tr_' + dataset_name + '.pdf', dpi=300, bbox_inches='tight', transparent=True)

# our_dsmn_t= [      74.20, 76.26, 78.62, 80.98, 83.10]  ## to refine.
# # our_dsmn=   [70.5, 72.67, 74.51, 76.97, 78.88, 80.22]  ## to refine.
# ########### 1/2/4/8/16 shot.
# coop =           [60.13, 62.71, 67.25, 70.62, 73.79]
# clip_adapter =   [69.27, 72.58, 75.37, 78.89, 81.79]
# tip_f =          [72.32, 75.29, 78.35, 80.69, 82.87]
# # zero_shot_clip = [66.43]  ## to refine


# # Plotting
# plt.figure(figsize=(4,3))
# plt.plot(shots[1:], dsmn_t_mean, '*-', label='DSMN-T (Ours)', color='red')
# # plt.plot(shots, our_dsmn, '*--', label='DSMN (Ours)', color='red')
# plt.plot(shots[1:], ape_t_mean, 'v-', label='APE-T', color='cyan')
# plt.plot(shots[1:], tip_f_mean, 'p-', label='TIP-Adapter-F', color='magenta')
# plt.plot(shots[1:], coop_mean, 'o-', label='CoOp', color='blue')
# plt.plot(shots[1:], clip_adapter_mean, 's-', label='CLIP-Adapter', color='black')
# # Plotting only points for CALIP and Zero-shot CLIP
# # plt.scatter(shots_calip_zero_shot, calip, marker='v', label='CALIP', color='purple')
# # plt.scatter(shots_calip_zero_shot, zero_shot_clip, marker='D', label='Zero-shot CLIP', color='brown')
# # Set labels, title, legend, etc.
# plt.xlabel('Shots Number')
# plt.ylabel('Accuracy (%)')
# plt.title('Average Results of 11 Datasets')
# plt.legend()
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.xticks([0, 1, 2, 4, 8, 16])
# max_num = max(dsmn_t_mean[-1], ape_t_mean[-1], tip_f_mean[-1])
# min_num = min(tip_f_mean[0], coop_mean[0], clip_adapter_mean[0])
# plt.ylim(min_num - (max_num - min_num) * 0.2)
# plt.tight_layout()
# # Show the plot
# # plt.show()
# plt.savefig(results_path+'resnet_tr_average.pdf', dpi=300, bbox_inches='tight', transparent=True)






# Data
# shots = [0, 1, 2, 4, 8, 16]
# shots_calip_zero_shot = [0]  # Only for CALIP and Zero-shot CLIP
# CoOp_results = [[57.6, 57.9, 59.8, 61.8, 63],  ## ImageNet
#                 [69.5, 78, 85.5, 90.5, 94.5],  ## Flowers..
#                 [44.1, 46, 54.1, 60, 64],  ## DTD
#                 [85.8, 82, 87, 85.8, 86.2],  ## pets
#                 [56.2, 58.2, 63, 67.8, 73.1], ##StanfordCars
#                 [62.9, 63.5, 69, 72.5, 76.2], ## UCF101
#                 [87.9, 87.92, 89.6, 90.1, 92.01], ## Caltech101
#                 [77.45, 77.8, 78.1, 78.5, 79.02], ## food
#                 [60, 60, 63.5, 65.8, 69.2], ## SUN
#                 [8.5, 18.5, 20.1, 27, 31.5], ## airplane
#                 [51.5, 60, 70.1, 77, 83], ## Eurosat
#                 ]
# coop_mean = get_datasets_mean(CoOp_results)
# ### copyed from: https://github.com/gaopengcuhk/Tip-Adapter/blob/main/exp.log
# Clip_adapter_results = [[61.20, 61.52, 61.84, 62.68, 63.59],  ## ImageNet
#                 [73.49, 81.61, 87.17, 91.72, 93.90],  ## Flowers..
#                 [45.80, 51.48, 56.86, 61.00, 65.96],  ## DTD
#                 [85.99, 86.73, 87.46, 87.65, 87.84],  ## pets
#                 [55.13, 58.74, 62.45, 67.89, 74.01], ##StanfordCars
#                 [62.20, 67.12, 69.05, 73.30, 76.76], ## UCF101
#                 [88.60, 89.37, 89.98, 91.40, 92.49], ## Caltech101
#                 [76.82, 77.22, 77.92, 78.04, 78.25], ## food
#                 [61.30, 63.29, 65.96, 67.50, 69.55], ## SUN
#                 [17.49, 20.10, 22.59, 26.25, 32.10], ## airplane
#                 [61.40, 63.90, 73.38, 77.93, 84.43], ## Eurosat
#                 ]
# clip_adapter_mean = get_datasets_mean(Clip_adapter_results)
# ### copyed from: https://github.com/gaopengcuhk/Tip-Adapter/blob/main/exp.log
# Tip_f_results = [[61.13, 61.69, 62.52, 64.00, 65.51],  ## ImageNet
#                 [64.87, 66.43, 70.55, 74.25, 78.03],  ## Flowers..
#                 [49.65, 53.72, 57.39, 62.71, 66.55],  ## DTD
#                 [87.00, 87.03, 87.54, 88.09, 89.70],  ## pets
#                 [58.86, 61.50, 64.57, 69.25, 75.74], ##StanfordCars
#                 [64.87, 66.43, 70.55, 74.25, 78.03], ## UCF101
#                 [89.33, 89.74, 90.56, 91.44, 92.86], ## Caltech101
#                 [77.51, 77.81, 78.24, 78.64, 79.43], ## food
#                 [62.50, 63.64, 66.21, 68.87, 71.47], ## SUN
#                 [20.22, 23.19, 25.80, 30.21, 35.55], ## airplane
#                 [59.53, 66.15, 74.12, 77.93, 84.54], ## Eurosat
#                 ]
# tip_f_mean = get_datasets_mean(Tip_f_results)
# ape_t_results =[[62.5, 63.2, 63.8, 64.6, 66.07],  ## ImageNet
#                 [83, 88, 92, 95, 96],  ## Flowers..
#                 [55, 59, 65.5, 67, 69.5],  ## DTD
#                 [87.2, 87.3, 88.5, 88.6, 90.2],  ## pets
#                 [60.5, 64, 68, 71, 77.5], ##StanfordCars
#                 [66.5, 69, 73.5, 75.5, 80], ## UCF101
#                 [90.5, 90.9, 91.7, 93, 93.2], ## Caltech101
#                 [77.6, 78.1, 78.6, 78.8, 79.5], ## food
#                 [66, 67.5, 69, 71, 73], ## SUN
#                 [25, 26, 29.5, 33.5, 40], ## airplane
#                 [65.5, 72, 75, 80.5, 87.5], ## Eurosat
#                 ]
# ape_t_mean = get_datasets_mean(ape_t_results)
#
# DSMN_T_results =[[63.64, 64.47, 64.96, 65.64, 66.75],  ## ImageNet
#                 [77.99, 84.73, 90.66, 93.75, 95.49],  ## Flowers..
#                 [58.63, 59.87, 63.24, 67.38, 70.04],  ## DTD
#                 [87.90, 88.77, 89.23, 89.53, 90.52],  ## pets
#                 [62.29, 64.93, 69.73, 74.12, 77.96], ##StanfordCars
#                 [68.57, 71.29, 74.44, 76.9, 80.23], ## UCF101
#                 [91.36, 91.81, 92.82, 93.35, 94.28], ## Caltech101
#                 [76.08, 76.2, 77, 77.42, 78.08], ## food
#                 [66.03, 67.09, 69.45, 71.64, 73.26], ## SUN
#                 [23.79, 24.75, 27.66, 31.35, 37.35], ## airplane
#                 [65.74, 66.26, 71.17, 77.07, 79.52], ## Eurosat
#                 ]
# dsmn_t_mean = get_datasets_mean(DSMN_T_results)
# zero_shot_clip_results = [60.32, 66.02, 42.20, 85.83, 55.74, 61.35, 85.92, 77.32, 58.52, 17.10, 37.52]
# for i in range(len(datasets)):
#     dataset_name = datasets[i]
#     our_dsmn_t = DSMN_T_results[i]
#     ape_t = ape_t_results[i]
#     tip_f = Tip_f_results[i]
#     coop = CoOp_results[i]
#     clip_adapter = Clip_adapter_results[i]
#     zero_shot_clip = [zero_shot_clip_results[i]]
#     # Plotting
#     plt.figure(figsize=(4,3))
#     plt.plot(shots[1:], our_dsmn_t, '*-', label='DSMN-T (Ours)', color='red')
#     # plt.plot(shots, our_dsmn, '*--', label='DSMN (Ours)', color='red')
#     plt.plot(shots[1:], ape_t, 'v-', label='APE-T', color='cyan')
#     plt.plot(shots[1:], tip_f, 'p-', label='TIP-Adapter-F', color='magenta')
#     plt.plot(shots[1:], coop, 'o-', label='CoOp', color='blue')
#     plt.plot(shots[1:], clip_adapter, 's-', label='CLIP-Adapter', color='black')
#     # Plotting only points for CALIP and Zero-shot CLIP
#     # plt.scatter(shots_calip_zero_shot, calip, marker='v', label='CALIP', color='purple')
#     # plt.scatter(shots_calip_zero_shot, zero_shot_clip, marker='D', label='Zero-shot CLIP', color='brown')
#     # Set labels, title, legend, etc.
#     plt.xlabel('Shots Number')
#     plt.ylabel('Accuracy (%)')
#     plt.title('Training-required Results on ' + dataset_name)
#     plt.legend()
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#     plt.xticks([0, 1, 2, 4, 8, 16])
#     max_num = max(our_dsmn_t[-1], ape_t[-1], tip_f[-1])
#     min_num = min(tip_f[0], coop[0], clip_adapter[0], coop[1])
#     plt.ylim(min_num - (max_num-min_num)*0.2)
#     plt.tight_layout()
#     # Show the plot
#     # plt.show()
#     plt.savefig(results_path+'resnet_tr_' + dataset_name + '.pdf', dpi=300, bbox_inches='tight', transparent=True)
#
# # our_dsmn_t= [      74.20, 76.26, 78.62, 80.98, 83.10]  ## to refine.
# # # our_dsmn=   [70.5, 72.67, 74.51, 76.97, 78.88, 80.22]  ## to refine.
# # ########### 1/2/4/8/16 shot.
# # coop =           [60.13, 62.71, 67.25, 70.62, 73.79]
# # clip_adapter =   [69.27, 72.58, 75.37, 78.89, 81.79]
# # tip_f =          [72.32, 75.29, 78.35, 80.69, 82.87]
# # # zero_shot_clip = [66.43]  ## to refine
#
#
# # Plotting
# plt.figure(figsize=(4,3))
# plt.plot(shots[1:], dsmn_t_mean, '*-', label='DSMN-T (Ours)', color='red')
# # plt.plot(shots, our_dsmn, '*--', label='DSMN (Ours)', color='red')
# plt.plot(shots[1:], ape_t_mean, 'v-', label='APE-T', color='cyan')
# plt.plot(shots[1:], tip_f_mean, 'p-', label='TIP-Adapter-F', color='magenta')
# plt.plot(shots[1:], coop_mean, 'o-', label='CoOp', color='blue')
# plt.plot(shots[1:], clip_adapter_mean, 's-', label='CLIP-Adapter', color='black')
# # Plotting only points for CALIP and Zero-shot CLIP
# # plt.scatter(shots_calip_zero_shot, calip, marker='v', label='CALIP', color='purple')
# # plt.scatter(shots_calip_zero_shot, zero_shot_clip, marker='D', label='Zero-shot CLIP', color='brown')
# # Set labels, title, legend, etc.
# plt.xlabel('Shots Number')
# plt.ylabel('Accuracy (%)')
# plt.title('Average Results of 11 Datasets')
# plt.legend()
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.xticks([0, 1, 2, 4, 8, 16])
# max_num = max(dsmn_t_mean[-1], ape_t_mean[-1], tip_f_mean[-1])
# min_num = min(tip_f_mean[0], coop_mean[0], clip_adapter_mean[0])
# plt.ylim(min_num - (max_num - min_num) * 0.2)
# plt.tight_layout()
# # Show the plot
# # plt.show()
# plt.savefig(results_path+'resnet_tr_average.pdf', dpi=300, bbox_inches='tight', transparent=True)