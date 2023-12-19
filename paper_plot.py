# ####################### accuracy vs GFloaps.
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Data
# methods = {
#     'Our APE-T': (0.005, 65.5, 100),
#     'Tip-Adapter-F': (0.1, 66, 80),
#     'Our APE': (0.004, 63.5, 90),
#     'CLIP-Adapter': (0.02, 64, 110),
#     'Tip-X': (1, 62.2, 70),
#     'Tip-Adapter': (0.00, 62, 75),
#     'CALIP': (0.00, 61.5, 85),
#     'CLIP': (0.00, 61, 95),
#     'CoOp': (0.07, 64.2, 105),
#     'CoCoOp': (0.08, 64, 100)
# }
#
# categories = {
#     'Zero-shot Methods': ['Tip-X', 'Tip-Adapter', 'CALIP', 'CLIP'],
#     'Non-prior Methods': ['CoOp', 'CoCoOp'],
#     'Prior-based Methods': ['Our APE-T', 'Tip-Adapter-F', 'Our APE', 'CLIP-Adapter']
# }
#
# colors = {
#     'Zero-shot Methods': 'red',
#     'Non-prior Methods': 'blue',
#     'Prior-based Methods': 'green'
# }
#
# markers = {
#     'Zero-shot Methods': '^',
#     'Non-prior Methods': 'o',
#     'Prior-based Methods': '*'
# }
#
# # Plot
# fig, ax = plt.subplots()
# for category, methods_list in categories.items():
#     for method in methods_list:
#         x, y, size = methods[method]
#         ax.scatter(x, y, label=method, color=colors[category], s=size, marker=markers[category])
#
# # Setting x-axis to logarithmic scale with custom ticks and limits
# ax.set_xscale('log')
# ax.set_xlim([0.1, 10])
# ax.set_xticks([0.001, 0.01, 0.1, 1, 10])
# ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
#
# # Labels and title
# ax.set_xlabel('GFLOPs')
# ax.set_ylabel('Accuracy (%)')
#
# # Legend for categories
# import matplotlib.lines as mlines
# legend_elements = [mlines.Line2D([0], [0], marker=markers[category], color='w', label=category,
#                   markersize=10, markerfacecolor=colors[category], markeredgewidth=0.5, markeredgecolor='black') for category in categories]
# ax.legend(handles=legend_elements, loc='best')
#
# # Annotate each point with its method name
# # for method, (x, y, size) in methods.items():
# #     ax.annotate(method, (x, y), fontsize=8, ha='right')
# for method, (x, y, size) in methods.items():
#     if method == 'Our APE-T':
#         ax.annotate(method, (x, y), fontsize=8, ha='right', color='red')
#     else:
#         ax.annotate(method, (x, y), fontsize=8, ha='right', color='black')
# plt.show()


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
results_path = '/Users/zhangyabin/Nutstore Files/我的坚果云/paper draft/CVPR2024/CLIP_tuning/images/'
# results_path = '/Users/zhangyabin/Nutstore Files/我的坚果云/paper draft/CVPR2024/'

###################### ablation of dynamic and static memory networks.
# # Data
# shots = [0, 1, 2, 4, 8, 16]
# shots_calip_zero_shot = [0]  # Only for CALIP and Zero-shot CLIP
# zero_shot_clip_results = [66.93, 69, 50, 90, 66, 70, 92, 84, 66, 22, 45]
# tf_only_static = [70.15, 70.38, 70.74, 71.47, 71.78]
# # tf_only_dynamic = [72.36, 72.41, 72.61, 72.88, 73.42]
# tf_dual_memory = [72.36, 72.41, 72.61, 72.88, 73.42]
#
# tr_only_static = [70.21, 71.30, 72.08, 72.80, 73.89]
# tr_only_dynamic = [72.27, 72.88, 73.47, 74.29, 75.21]
# tr_dual_memory = [72.28, 72.97, 73.6,  74.39, 75.34]
#
# clip = [69.93]
# DMN_ZS = [72.25]
# # zero_shot_clip = [zero_shot_clip_results[i]]
# # Plotting
# plt.figure(figsize=(5.4, 3.6))
# # ipdb.set_trace()
# plt.plot(shots[1:], tr_dual_memory, '*-', label='DMN', color='red')
# plt.plot(shots[1:], tr_only_static, '*--', label='SMN', color='red')
# plt.plot(shots[1:], tf_dual_memory, 'v-', label='DMN-TF', color='blue')
# plt.plot(shots[1:], tf_only_static, 'v--', label='SMN-TF', color='blue')
# # plt.plot(shots[1:], clip, 'o-.', label='CLIP', color='black')
# # plt.axhline(y=69.93, color='black', linestyle='-.', label='CLIP')
# # Plotting only points for CALIP and Zero-shot CLIP
# plt.scatter(shots_calip_zero_shot, DMN_ZS, marker='s', label='DMN-ZS', color='purple')
# plt.scatter(shots_calip_zero_shot, clip, marker='D', label='Zero-shot CLIP', color='black')
# # Set labels, title, legend, etc.
# plt.xlabel('Shots Number')
# plt.ylabel('Acc. (%)')
# # plt.title('Dynamic and Static Memories')
# plt.legend()
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.xticks([0, 1, 2, 4, 8, 16])
# # max_num = max(our_dsmn_t[-1], promptsrc[-1], maple[-1])
# # min_num = min(zero_shot_clip[0], cocoop[0], coop[0], maple[0])
# plt.ylim(69, 77.5)
# plt.tight_layout()
# # Show the plot
# # plt.show()
# plt.savefig(results_path+'memory_ablation.pdf', dpi=300, bbox_inches='tight', transparent=True)
############ ablation of dynamic and static memory networks, only TF results.
# Data
shots = [0, 1, 2, 4, 8, 16]
shots_calip_zero_shot = [0]  # Only for CALIP and Zero-shot CLIP
tf_only_static = [70.15, 70.38, 70.74, 71.47, 71.78]  ## 71.78-->72.08
tf_only_dynamic = [72.30, 72.36, 72.44, 72.61, 72.91]
tf_dual_memory = [72.36, 72.41, 72.61, 72.88, 73.42]

# tr_only_static = [70.21, 71.30, 72.08, 72.80, 73.89]
# tr_only_dynamic = [72.27, 72.88, 73.47, 74.29, 75.21]
# tr_dual_memory = [72.28, 72.97, 73.6,  74.39, 75.34]

clip = [69.93]
DMN_ZS = [72.25]
# Plotting
plt.figure(figsize=(3.6, 2.7))
# ipdb.set_trace()
plt.plot(shots[1:], tf_dual_memory, 'v-', label='Dual Memory Networks', color='red')
plt.plot(shots[1:], tf_only_dynamic, '*-', label='Only Dynamic Memory Network', color='orange')
plt.plot(shots[1:], tf_only_static, 'o-', label='Only Static Memory Network', color='blue')

# plt.plot(shots[1:], tf_only_static, 'v--', label='SMN-TF', color='blue')
# plt.plot(shots[1:], clip, 'o-.', label='CLIP', color='black')
# plt.axhline(y=69.93, color='black', linestyle='-.', label='CLIP')
# Plotting only points for CALIP and Zero-shot CLIP
# plt.scatter(shots_calip_zero_shot, DMN_ZS, marker='s', label='DMN-ZS', color='purple')
plt.scatter(shots_calip_zero_shot, clip, marker='D', label='Zero-shot CLIP', color='black')
# Set labels, title, legend, etc.
plt.xlabel('Shots Number')
plt.ylabel('Acc. (%)')
# plt.title('Dynamic and Static Memories')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks([0, 1, 2, 4, 8, 16])
# max_num = max(our_dsmn_t[-1], promptsrc[-1], maple[-1])
# min_num = min(zero_shot_clip[0], cocoop[0], coop[0], maple[0])
plt.ylim(69.5, 77)
plt.tight_layout()
plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
# Show the plot
# plt.show()
plt.savefig(results_path+'memory_ablation.pdf', dpi=300, bbox_inches='tight', transparent=True)


###################### analyses on the memory length.
length = [1,3,5,10,30,50,70]
DMN_ZS = [70.23, 70.92, 71.34, 71.84, 72.22, 72.25, 72.25]
clip = [69.93]

# zero_shot_clip = [zero_shot_clip_results[i]]
# Plotting
plt.figure(figsize=(3.6, 2.7))
plt.plot(length, DMN_ZS, '*-', label='DMN-ZS', color='red')
plt.scatter(shots_calip_zero_shot, clip, marker='D', label='Zero-shot CLIP', color='black')
# Set labels, title, legend, etc.
# plt.xlabel('Memory Length $L$')
plt.ylabel('Acc. (%)')
# plt.title('Dynamic and Static Memories')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks([1, 5,10,30,50,70])
# max_num = max(our_dsmn_t[-1], promptsrc[-1], maple[-1])
# min_num = min(zero_shot_clip[0], cocoop[0], coop[0], maple[0])
# plt.ylim(69, 77.5)
plt.tight_layout()
# Show the plot
# plt.show()
plt.savefig(results_path+'memory_length.pdf', dpi=300, bbox_inches='tight', transparent=True)


###################### analyses on the beta
beta = [0.5, 1.5, 3.5, 5.5, 7.5, 10.5]
DMN_TF = [72.20, 72.46, 73.14, 73.42, 73.34, 73.18]  ## 16-shot
# clip = [69.93]

# zero_shot_clip = [zero_shot_clip_results[i]]
# Plotting
plt.figure(figsize=(3.6, 2.7))
plt.plot(beta, DMN_TF, '*-', label='DMN-TF', color='red')
# plt.scatter(shots_calip_zero_shot, clip, marker='D', label='Zero-shot CLIP', color='black')
# Set labels, title, legend, etc.
# plt.xlabel('Values of β')
plt.ylabel('Acc. (%)')
# plt.title('Dynamic and Static Memories')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks([0.5, 1.5, 3.5, 5.5, 7.5, 10.5])
# plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
# max_num = max(our_dsmn_t[-1], promptsrc[-1], maple[-1])
# min_num = min(zero_shot_clip[0], cocoop[0], coop[0], maple[0])
plt.ylim(71.8, 73.5)
plt.tight_layout()
# Show the plot
# plt.show()
plt.savefig(results_path+'DMN_TF_various_beta.pdf', dpi=300, bbox_inches='tight', transparent=True)


####################### analyses on position of projection layers.
categories = ['Q',    'K',   'V',   'QKV',   'O',   'QKVO']
# values =     [73.29, 73.46, 74.43, 74.47,  75.29,  75.39]  ### all use bias
values =     [73.37, 73.36, 74.29,  74.58,  75.29,  75.39]   ### QKV affine, output bias
#################################### 74.28           74.64  ## shared.
###### 可能VO 是最优解，试一下只在这两个上做:  74.88.
# 绘制柱状图
colors = ['#FF8C00']
plt.figure(figsize=(3.6, 2.7))
plt.bar(categories, values, color=colors)
# 设置标题和坐标轴标签
# plt.title('Position of Projection Layers')
plt.ylabel('Acc. (%)')
plt.ylim(73, 76)
plt.tight_layout()
plt.savefig(results_path+'projection_position.pdf', dpi=300, bbox_inches='tight', transparent=True)




# DMN-results with fixed classifier weights.
###########################
shots = [1, 2, 4, 8, 16]
searched =  [75.45, 77.13, 79.32, 81.46, 83.49]  ## 71.78-->72.08
fixed =     [74.32, 76.28, 78.97, 81.12, 83.21]
PromptSRC = [72.32, 75.29, 78.34, 80.69, 82.87]
# Plotting
plt.figure(figsize=(3.6, 2.7))
# ipdb.set_trace()
plt.plot(shots, searched, '*-', label='DMN-Searched', color='red')
plt.plot(shots, fixed, 'v-', label='DMN-Fixed', color='orange')
plt.plot(shots, PromptSRC, 'o-', label='PromptSRC', color='blue')

# Set labels, title, legend, etc.
plt.xlabel('Shots Number')
plt.ylabel('Acc. (%)')
# plt.title('Dynamic and Static Memories')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks([1, 2, 4, 8, 16])
# max_num = max(our_dsmn_t[-1], promptsrc[-1], maple[-1])
# min_num = min(zero_shot_clip[0], cocoop[0], coop[0], maple[0])
# plt.ylim(72,74)
plt.tight_layout()
# plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
# Show the plot
# plt.show()
plt.savefig(results_path+'searched_fixed_weights.pdf', dpi=300, bbox_inches='tight', transparent=True)



################### softmax_ours
shots = [0, 1, 2, 4, 8, 16]
softmax = [72.33, 72.35, 72.53, 72.78, 73.15]  ## 71.78-->72.08
ours = [72.36, 72.41, 72.61, 72.88, 73.42]
# Plotting
plt.figure(figsize=(3.6, 2.7))
# ipdb.set_trace()
plt.plot(shots[1:], ours, '*-', label='Ours', color='red')
plt.plot(shots[1:], softmax, 'v-', label='SoftMax', color='blue')


# Set labels, title, legend, etc.
plt.xlabel('Shots Number')
plt.ylabel('Acc. (%)')
# plt.title('Dynamic and Static Memories')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks([1, 2, 4, 8, 16])
plt.ylim(72,74)
plt.tight_layout()
# plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
# Show the plot
# plt.show()
plt.savefig(results_path+'softmax_ours.pdf', dpi=300, bbox_inches='tight', transparent=True)