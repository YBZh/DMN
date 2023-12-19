## few-shot results, training required, vit backbone.

######## results of tip, tip-f, and clip-adapter can be achieved from   https://github.com/gaopengcuhk/Tip-Adapter/blob/main/exp.log

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ipdb
# datasets = ['ImageNet', 'Flowers102', 'DTD', 'OxfordPets', 'StanfordCars', 'UCF101', 'Caltech101', 'Food101', 'SUN397', 'FGVCAircraft', 'EuroSAT']
# methods = ['CoOp', 'CoCoOp', 'MaPLe', 'PromptSRC', 'DSMN', 'DSMN_T']
# results_path = '/Users/zhangyabin/Nutstore Files/我的坚果云/paper draft/CVPR2024/images/results_path_vit_training_required/'
datasets = ['ImageNet', 'Flowers102', 'DTD', 'OxfordPets', 'StanfordCars', 'UCF101', 'Caltech101', 'Food101', 'SUN397', 'FGVCAircraft', 'EuroSAT', 'Mean']
methods = ['CoOp', 'CoCoOp', 'MaPLe', 'PromptSRC', 'DSMN', 'DSMN-T']
excel_path = '/Users/zhangyabin/Nutstore Files/我的坚果云/paper draft/CVPR2024/training_required_vit.xlsx'
results_path = '/Users/zhangyabin/Nutstore Files/我的坚果云/paper draft/CVPR2024/CLIP_tuning/images/results_path_vit_training_required/'
# results_path = '/Users/zhangyabin/Nutstore Files/我的坚果云/paper draft/CVPR2024/'

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

# Data
shots = [0, 1, 2, 4, 8, 16]
shots_calip_zero_shot = [0]  # Only for CALIP and Zero-shot CLIP
# CoOp_results = [[66.33, 67.07, 68.73, 70.63, 71.87],  ## ImageNet
#                 [77.53, 87.33, 92.17, 94.97, 97.07],  ## Flowers..
#                 [50.23, 53.60, 58.70, 64.77, 69.87],  ## DTD
#                 [90.37, 89.80, 92.57, 91.27, 91.87],  ## pets
#                 [67.43, 70.50, 74.47, 79.30, 83.07], ##StanfordCars
#                 [71.23, 73.43, 77.10, 80.20, 82.23], ## UCF101
#                 [92.60, 93.07, 94.40, 94.37, 95.57], ## Caltech101
#                 [84.33, 84.40, 84.47, 82.67, 84.20], ## food
#                 [66.77, 66.53, 69.97, 71.53, 74.67], ## SUN
#                 [21.37, 26.20, 30.83, 39.00, 43.40], ## airplane
#                 [54.93, 65.17, 70.80, 78.07, 84.93], ## Eurosat
#                 ]
# CoCoOp_results = [[69.43, 69.78, 70.39, 70.63, 70.83],
#                   [72.08, 75.79, 78.40, 84.30, 87.84],
#                   [48.54, 52.17, 55.04, 58.89, 63.04],
#                   [91.27, 92.64, 92.81, 93.45, 93.34],
#                   [67.22, 68.37, 69.39, 70.44, 71.57],
#                   [70.30, 73.51, 74.82, 77.14, 78.14],
#                   [93.83, 94.82, 94.98, 95.04, 95.16],
#                   [85.65, 86.22, 86.88, 86.97, 87.25],
#                   [68.33, 69.03, 70.21, 70.84, 72.15],
#                   [12.68, 15.06, 24.79, 26.61, 31.21],
#                   [55.33, 46.74, 65.56, 68.21, 73.22],
#                   ]
# MaPLe_results =  [[62.67, 65.10, 67.70, 70.30, 72.33],
#                   [83.30, 88.93, 92.67, 95.80, 97.00],
#                   [52.13, 55.50, 61.00, 66.50, 71.33],
#                   [89.10, 90.87, 91.90, 92.57, 92.83],
#                   [66.60, 71.60, 75.30, 79.47, 83.57],
#                   [71.83, 74.60, 78.47, 81.37, 85.03],
#                   [92.57, 93.97, 94.43, 95.20, 96.00],
#                   [80.50, 81.47, 81.77, 83.60, 85.33],
#                   [64.77, 67.10, 70.67, 73.23, 75.53],
#                   [26.73, 30.90, 34.87, 42.00, 48.40],
#                   [71.80, 78.30, 84.50, 87.73, 92.33],
#                   ]
# PromptSRC_results = [[68.13, 69.77, 71.07, 72.33, 73.17],
#                      [85.93, 91.17, 93.87, 96.27, 97.60],
#                      [56.23, 59.97, 65.53, 69.87, 72.73],
#                      [92.00, 92.50, 93.43, 93.50, 93.67],
#                      [69.40, 73.40, 77.13, 80.97, 83.83],
#                      [74.80, 78.50, 81.57, 84.30, 86.47],
#                      [93.67, 94.53, 95.27, 95.67, 96.07],
#                      [84.87, 85.70, 86.17, 86.90, 87.50],
#                      [69.67, 71.60, 74.00, 75.73, 77.23],
#                      [27.67, 31.70, 37.47, 43.27, 50.83],
#                      [73.13, 79.37, 86.30, 88.80, 92.43]
#                      ]
# DSMN_results =   [[72.25, 72.36, 72.41, 72.61, 72.88, 73.42],
#                   [74.79, 85.22, 92.29, 95.09, 96.63, 96.91],
#                   [55.85, 58.27, 61.29, 65.01, 68.32, 71.93],
#                   [92.04, 91.96, 91.96, 92.04, 92.20, 92.86],
#                   [67.96, 68.96, 70.69, 73.95, 76.79, 78.58],
#                   [72.51, 74.28, 76.66, 79.17, 82.18, 83.08],
#                   [95.38, 95.62, 95.74, 96.02, 96.06, 96.47],
#                   [85.08, 85.04, 85.19, 85.29, 85.36, 85.95],
#                   [70.18, 70.48, 71.41, 72.87, 74.42, 75.83],
#                   [30.03, 30.75, 32.97, 37.32, 41.52, 44.67],
#                   [59.43, 66.41, 69.02, 77.27, 81.26, 82.70],
#                   ]
# DSMN_T_results = [[72.25, 72.30, 72.99, 73.65, 74.40, 75.32],
#                   [74.79, 87.54, 92.81, 95.49, 97.52, 97.97],
#                   [55.85, 60.87, 63.53, 67.55, 70.39, 74.47],
#                   [92.04, 92.42, 92.61, 93.02, 93.24, 94.03],
#                   [67.96, 71.72, 73.92, 77.80, 80.92, 83.87],
#                   [72.51, 77.53, 80.57, 83.43, 85.06, 86.62],
#                   [95.38, 95.62, 95.82, 96.75, 96.92, 97.28],
#                   [85.08, 85.00, 85.05, 85.38, 85.82, 86.67],
#                   [70.18, 72.34, 73.58, 75.03, 76.66, 78.41],
#                   [30.03, 32.04, 35.55, 38.58, 42.87, 49.02],
#                   [59.43, 68.83, 72.51, 78.14, 81.56, 83.86],
#                   ]
zero_shot_clip_results = [66.93, 69, 50, 90, 66, 70, 92, 84, 66, 22, 45]
for i in range(len(datasets)):
    dataset_name = datasets[i]
    our_dsmn = all_cate['DSMN'][i]
    our_dsmn_t = all_cate['DSMN-T'][i]
    promptsrc = all_cate['PromptSRC'][i]
    maple = all_cate['MaPLe'][i]
    coop = all_cate['CoOp'][i]
    cocoop = all_cate['CoCoOp'][i]
    # zero_shot_clip = [zero_shot_clip_results[i]]
    # Plotting
    plt.figure(figsize=(4,3))
    # ipdb.set_trace()
    plt.plot(shots[1:], our_dsmn_t, '*-', label='DMN (Ours)', color='red')
    plt.plot(shots[1:], our_dsmn[1:], '*--', label='DMN-TF (Ours)', color='red')
    plt.plot(shots[1:], promptsrc, 'v-', label='PromptSRC', color='cyan')
    plt.plot(shots[1:], maple, 'p-', label='MaPLe', color='magenta')
    plt.plot(shots[1:], coop, 'o-', label='CoOp', color='blue')
    plt.plot(shots[1:], cocoop, 's-', label='CoCoOp', color='black')
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
    # max_num = max(our_dsmn_t[-1], promptsrc[-1], maple[-1])
    # min_num = min(zero_shot_clip[0], cocoop[0], coop[0], maple[0])
    # plt.ylim(min_num - (max_num-min_num)*0.2)
    plt.tight_layout()
    # Show the plot
    # plt.show()
    plt.savefig(results_path+'vit_tr_' + dataset_name + '.pdf', dpi=300, bbox_inches='tight', transparent=True)

# our_dsmn_t= [      74.20, 76.26, 78.62, 80.98, 83.10]  ## to refine.
# our_dsmn=   [70.5, 72.67, 74.51, 76.97, 78.88, 80.22]  ## to refine.
# ########### 1/2/4/8/16 shot.
# coop =           [67.56, 70.65, 74.02, 76.98, 79.89]
# cocoop =         [66.79, 67.65, 71.21, 72.96, 74.90]
# maple =          [69.27, 72.58, 75.37, 78.89, 81.79]
# promptsrc =      [72.32, 75.29, 78.35, 80.69, 82.87]
# zero_shot_clip = [66.43]  ## to refine
#
# # Plotting
# plt.figure(figsize=(4,3))
# plt.plot(shots[1:], our_dsmn_t, '*-', label='DSMN-T (Ours)', color='red')
# plt.plot(shots, our_dsmn, '*--', label='DSMN (Ours)', color='red')
# plt.plot(shots[1:], promptsrc, 'v-', label='PromptSRC', color='cyan')
# plt.plot(shots[1:], maple, 'p-', label='MaPLe', color='magenta')
# plt.plot(shots[1:], coop, 'o-', label='CoOp', color='blue')
# plt.plot(shots[1:], cocoop, 's-', label='CoCoOp', color='black')
# # Plotting only points for CALIP and Zero-shot CLIP
# # plt.scatter(shots_calip_zero_shot, calip, marker='v', label='CALIP', color='purple')
# plt.scatter(shots_calip_zero_shot, zero_shot_clip, marker='D', label='Zero-shot CLIP', color='brown')
# # Set labels, title, legend, etc.
# plt.xlabel('Shots Number')
# plt.ylabel('Accuracy (%)')
# plt.title('Average Results of 11 Datasets')
# plt.legend()
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.xticks([0, 1, 2, 4, 8, 16])
# max_num = max(our_dsmn_t[-1], promptsrc[-1], maple[-1])
# min_num = min(zero_shot_clip[0], cocoop[0], coop[0], maple[0])
# plt.ylim(min_num - (max_num - min_num) * 0.2)
# plt.tight_layout()
# # Show the plot
# # plt.show()
# plt.savefig(results_path+'vit_tr_average.pdf', dpi=300, bbox_inches='tight', transparent=True)