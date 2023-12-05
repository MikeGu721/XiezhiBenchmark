'''
Given:
option_number
random_seed
-->each model performance [Done]
-->model parameter performance

画图
1. result_log --> each model performance
2. result_log --> model parameter performance
3. result_log --> traditional method model performance
4. result_log --> model performance with different option number
5. 【获取model release time】 result_log --> model release time performance
'''

import os
import json
from conclude import *


result_log_dir = './new_result'
hitk = 1
model_performance = {}

for file in os.listdir(result_log_dir):
    model_name = '/'.join(file.split('_')[0].split('--')[1:3])
    option_number = file.split('_')[-2]
    few_shot_num = file.split('_')[-3]
    random_seed = file.split('_')[-1].strip('.json')
    if 'xiezhi' in file:
        dataset = '_'.join(file.split('_')[-6:-3])
    else:
        dataset = file.split('_')[-4]

    if model_name not in model_performance:
        model_performance[model_name] = {}

    json_file = os.path.join(result_log_dir, file)
    label2result = get_evaluation_result_from_check_path(json_file)
    if 'OVERALL' not in label2result: continue
    label2hitk = get_hitk_result(label2result, hitk)

    if dataset not in model_performance[model_name]:
        model_performance[model_name][dataset] = {}

    if few_shot_num not in model_performance[model_name][dataset]:
        model_performance[model_name][dataset][few_shot_num] = {}

    model_performance[model_name][dataset][few_shot_num][option_number] = label2hitk['OVERALL']


see_option = '50'

for model_name in model_performance:
    print()
    for dataset in model_performance[model_name]:
        for few_shot_num in model_performance[model_name][dataset]:
            if see_option in model_performance[model_name][dataset][few_shot_num]:
                print(model_name, dataset, few_shot_num, model_performance[model_name][dataset][few_shot_num][see_option])

