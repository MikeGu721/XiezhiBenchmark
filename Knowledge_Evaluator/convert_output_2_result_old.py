import collections
import json
import os
from conclude import *

import matplotlib.pyplot as plt
import seaborn as sns
hitk = 1
output_path = './output_results/check_result'

random_seed = 42
lora_name = None

model_performance = collections.defaultdict(dict)

for model_name in ['THUDM/chatglm2-6b', 'THUDM/chatglm3-6b']:
    for benchmark_name in ['xiezhi_inter_eng']:
        perf = []
        for few_shot_num in [0, 1, 2, 3]:
            for option_num in [4, 8, 16, 32, 50]:

                UID = '%s_%s_%s_%s_%s_%s' % (
                model_name.replace('/', ''), lora_name, benchmark_name, few_shot_num, option_num, random_seed)

                json_file = os.path.join(output_path, UID + '.json')
                if not os.path.exists(json_file): continue
                label2result = get_evaluation_result_from_check_path(json_file)
                if 'OVERALL' not in label2result: continue
                label2hitk = get_hitk_result(label2result, hitk)
                if few_shot_num not in model_performance[model_name]:
                    model_performance[model_name][few_shot_num] = {}
                model_performance[model_name][few_shot_num][option_num] = label2hitk['OVERALL']
                perf.append(float('%.7f'%label2hitk['OVERALL']['mean']))
        print(model_name,perf)
        print(len(perf))
        if len(perf) != 5 * 4:
            print(len(perf))
            continue
        perf = np.array(perf)
        print(model_name,perf.reshape(4, -1))
        sns.heatmap(perf.reshape(4, -1), center=0)
        plt.show()
        plt.clf()
