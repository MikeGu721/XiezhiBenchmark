import collections
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pandas.tseries.offsets import DateOffset

def get_evaluation_result_from_check_path(check_path):
    f = open(check_path, encoding='utf-8')
    label2result = collections.defaultdict(list)
    for line in f:
        jsonline = json.loads(line)
        answer_rank = np.argsort(jsonline['options']).tolist().index(int(jsonline['answer']))
        for label in jsonline['labels']:
            label2result[label].append(answer_rank)
        label2result['OVERALL'].append(answer_rank)
    return label2result


def get_hitk_result(label2result, k):
    label2hitk = {}
    for label in label2result:
        cor = [1  if i < k else 0 for i in label2result[label]]
        label2hitk[label] = {'mean':np.mean(cor),'std':np.std(cor),'num':len(cor)}
    return label2hitk

def get_mrr_result(label2result):
    label2mrr = {}
    for label in label2result:
        mrr_label = [1/(i+1) for i in label2result[label]]
        label2mrr[label] = {'mean':np.mean(mrr_label), 'std':np.std(mrr_label), 'num':len(mrr_label)}
    return label2mrr

def draw_parameter_performance_figure(parameter2performance:dict):
    parameter2performance = {'bloomz-1b1': {'parameter': 1000, 'performance': 0.1},
                             'bloomz-1b7': {'parameter': 1000, 'performance': 0.11},
                             'llama-2b': {'parameter': 2000, 'performance': 0.15},
                             'llama2-2b': {'parameter': 2000, 'performance': 0.16},
                             'bloomz-10b': {'parameter': 100000, 'performance': 0.19},
                             'llama-10b': {'parameter': 100000, 'performance': 0.2},
                             'bloomz-100b': {'parameter': 10000000, 'performance': 0.24},
                             'llama-100b': {'parameter': 10000000, 'performance': 0.25},
                             'gpt4': {'parameter': 100000000, 'performance': 0.35},
                             'gpt4-turbo': {'parameter': 100000000, 'performance': 0.39},
                             'gpt4-single': {'parameter': 100000000, 'performance': 0.41},
                             'gpt4-tturbo': {'parameter': 100000000, 'performance': 0.42}}
    # 将字典转换为两个列表
    x = []
    y = []
    labels = []
    for model_name, values in parameter2performance.items():
        parameter_number = values['parameter']
        performance = values['performance']
        x.append(parameter_number)
        y.append(performance)
        labels.append(model_name)

    # 创建散点图
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=np.log10(x), y=y)

    # 设置x轴的刻度和刻度标签
    x_ticks = np.arange(3, 9)
    plt.xticks(x_ticks, 10 ** x_ticks)

    # 设置x轴和y轴的标签
    plt.xlabel('Parameter')
    plt.ylabel('Performance')

    # 在每个点上添加文本
    for i in range(len(x)):
        plt.text(np.log10(x[i]), y[i], labels[i])

    # 显示图形
    plt.show()

def draw_releasetime_performance_figure(parameter2performance:dict):
    # 数据
    parameter2performance = {'bloomz-1b1': {'releasetime': '2023-12-01', 'performance': 0.1},
                             'bloomz-1b7': {'releasetime': '2023-12-08', 'performance': 0.11},
                             'llama-2b': {'releasetime': '2023-12-29', 'performance': 0.15},
                             'llama2-2b': {'releasetime': '2024-05-01', 'performance': 0.16},
                             'bloomz-10b': {'releasetime': '2024-07-01', 'performance': 0.19},
                             'llama-10b': {'releasetime': '2024-07-09', 'performance': 0.2},
                             'bloomz-100b': {'releasetime': '2024-07-30', 'performance': 0.24},
                             'llama-100b': {'releasetime': '2024-09-11', 'performance': 0.25},
                             'gpt4': {'releasetime': '2024-09-30', 'performance': 0.35},
                             'gpt4-turbo': {'releasetime': '2024-11-25', 'performance': 0.39},
                             'gpt4-single': {'releasetime': '2025-03-15', 'performance': 0.41},
                             'gpt4-tturbo': {'releasetime': '2025-03-16', 'performance': 0.42}}

    # 将字典转换为DataFrame
    df = pd.DataFrame(parameter2performance).T
    df['releasetime'] = pd.to_datetime(df['releasetime'])
    df = df.sort_values('releasetime')

    # 创建日期刻度
    start = df['releasetime'].min() - DateOffset(months=3)
    end = df['releasetime'].max() + DateOffset(months=3)
    date_ticks = pd.date_range(start=start, end=end, freq='Q')

    # 创建散点图
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df['releasetime'], y=df['performance'])

    # 设置x轴的刻度和刻度标签
    plt.xticks(date_ticks, date_ticks.strftime('%Y-%m'), rotation=45)

    # 设置x轴和y轴的标签
    plt.xlabel('Release Time')
    plt.ylabel('Performance')

    # 在每个点上添加文本
    for i in range(len(df)):
        plt.text(df['releasetime'].iloc[i], df['performance'].iloc[i], df.index[i])

    # 显示图形
    plt.show()


if __name__ == '__main__':
    # draw_parameter_performance_figure({})
    draw_releasetime_performance_figure({})
