import os
import argparse
import pandas as pd
import torch
import json
from evaluators.moss import Moss_Evaluator
from evaluators.chatglm import ChatGLM_Evaluator
from evaluators.minimax import MiniMax_Evaluator

import time
choices = ["A", "B", "C", "D"]

def main(args):
    if "moss" in args.model_name:
        evaluator=Moss_Evaluator(
            choices=choices,
            k=args.ntrain,
            model_name=args.model_name
        )
    elif "chatglm" in args.model_name:
        if args.cuda_device:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
        device = torch.device("cuda")
        evaluator=ChatGLM_Evaluator(
            choices=choices,
            k=args.ntrain,
            model_name=args.model_name,
            device=device
        )
    elif "minimax" in args.model_name:
        evaluator=MiniMax_Evaluator(
            choices=choices,
            k=args.ntrain,
            group_id=args.minimax_group_id,
            api_key=args.minimax_key,
            model_name=args.model_name
        )
    else:
        print("Unknown model name")
        return -1

    subject_name=args.task
    if not os.path.exists(r"logs"):
        os.mkdir(r"logs")
    run_date=time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
    save_result_dir=os.path.join(r"logs",f"{args.model_name}_{run_date}")
    os.mkdir(save_result_dir)
    print(subject_name)
    
    var_file_path=os.path.join('tasks',f'{subject_name}', 'xiezhi.v1.json')
    jsonfile = open(var_file_path, encoding='utf-8')

    # 初始化DataFrame
    var_df = pd.DataFrame(columns=['id', 'question', 'A', 'B', 'C', 'D', 'answer', 'explanation'])

    # 解析JSON数据，逐行添加到DataFrame中
    idx = 0
    for jsonline in jsonfile:
        jsonline = json.loads(jsonline)
        options = jsonline['options'].split('\n')
        for o in options:
            o = str(o).strip().strip(' "\'').strip('"').rstrip('"')
        if(len(options) < 4):
            print("Error: options length is not 4 at line ", idx)
            continue
        var_df.loc[idx] = [
            idx,  # id自动递增编号
            jsonline['question'],
            options[0],
            options[1],
            options[2],
            options[3],
            get_option(jsonline['answer'].strip().strip(' "\'').strip('"').rstrip('"'), options),  # 将答案转换为选项号，并存储在answer列中
            ' '  # 将标签以逗号分隔，并存储在explanation列中
        ]
        idx += 1
        
    sample_num = args.sample_num
    var_df = var_df.sample(n=min(sample_num, idx))
    correct_ratio = evaluator.eval_subject(subject_name, var_df, few_shot=args.few_shot,save_result_dir=save_result_dir)
    print("Acc:",correct_ratio)


def get_option(s, options):
    if s == options[0]:
        return 'A'
    elif s == options[1]:
        return 'B'
    elif s == options[2]:
        return 'C'
    elif s == options[3]:
        return 'D'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--minimax_group_id", type=str,default="xxx") # for minimax
    parser.add_argument("--minimax_key", type=str,default="xxx") # for minimax
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--model_name",type=str)
    parser.add_argument("--task","-t",type=str,default="xiezhi_inter_chn")
    parser.add_argument("--cuda_device", type=str) # for chatglm
    parser.add_argument("--sample_num", "-s", type=int, default=10)
    args = parser.parse_args()
    main(args)