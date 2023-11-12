import collections
import json
import numpy as np

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

