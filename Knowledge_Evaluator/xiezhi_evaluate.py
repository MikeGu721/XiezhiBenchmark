import json
import os.path
import random
import time
import pandas
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer, LlamaTokenizer, AutoModelForCausalLM, GenerationConfig, BloomForCausalLM, \
    AutoModel, LlamaForCausalLM, GPTNeoXForCausalLM, AutoConfig
from collections import defaultdict
import transformers
import numpy as np
import tqdm
import torch.nn as nn


def get_names(model_name, lora_name, model_cache_dir):
    try:
        name = 'models--%s--%s' % (model_name.split('/')[0], model_name.split('/')[1])
        input_model_name = os.path.join(model_cache_dir, name, 'snapshots')
        input_model_name = os.path.join(input_model_name, os.listdir(input_model_name)[0])
    except:
        input_model_name = None

    input_lora_name = None
    try:
        if lora_name and '/' in lora_name:
            name = 'models--%s--%s' % (lora_name.split('/')[0], lora_name.split('/')[1])
            input_lora_name = os.path.join(lora_cache_dir, name, 'snapshots')
            input_lora_name = os.path.join(input_lora_name, os.listdir(input_lora_name)[0])
    except:
        input_lora_name = None
    return input_model_name, input_lora_name

# def get_causalLM(model_name, model_cache_dir, lora_name, lora_cache_dir, load_in_8bit=False):
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype=torch.float16,
#         cache_dir=model_cache_dir,
#         trust_remote_code=True,
#         device_map = 'auto'
#     )
#     model.eval()
#     tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache_dir, trust_remote_code=True)
#     return model, tokenizer

def get_causalLM(model_name, model_cache_dir, lora_name, lora_cache_dir, load_in_8bit=False):
    causalLM = \
        LlamaForCausalLM if 'llama' in model_name.lower() or 'cute' in model_name.lower() else \
            BloomForCausalLM if 'bloom' in model_name.lower() or 'belle' in model_name.lower() else \
                AutoModel if 'glm' in model_name.lower() else \
                    GPTNeoXForCausalLM if 'pythia' in model_name.lower() else \
                        AutoModelForCausalLM

    device_map = 'balanced_low_0'
    if model_cache_dir:
        model = causalLM.from_pretrained(model_name, cache_dir=model_cache_dir,
                                         torch_dtype=torch.float16, device_map='auto',
                                         load_in_8bit=load_in_8bit, trust_remote_code=True)
    else:
        model = causalLM.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir=model_cache_dir,
                                         device_map='auto', load_in_8bit=load_in_8bit,
                                         trust_remote_code=True)
    if 'none' in str(lora_name).lower(): lora_name = None
    if 'none' in str(lora_cache_dir).lower(): lora_cache_dir = None
    if lora_name and lora_cache_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_name, cache_dir=lora_cache_dir, device_map='auto',
                                          torch_dtype=torch.float16, )
    elif lora_name:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_name, device_map='auto', cache_dir='../LLM/CACHE_DIR',
                                          torch_dtype=torch.float16, )

    model.half()
    model.eval()
    if 'llama' not in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=model_cache_dir)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=model_cache_dir)
    tokenizer.padding_side = 'left'

    if 'chatglm' not in model_name.lower():
        model.config.pad_token_id = tokenizer.pad_token_id = 0
        model.config.bos_token_id = tokenizer.bos_token_id = 1
        model.config.eos_token_id = tokenizer.eos_token_id = 2
    return model, tokenizer


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class TestDataset(Dataset):
    def __init__(self, ids=None, data=None,answers=None,labels=None):
        super().__init__()
        self.ids, self.input, self.answer, self.labels = [], [], [], []
        if ids:
            self.extend(ids, data, answers, labels)

    def append(self, id, data, answer,label):
        self.ids.append(id)
        self.input.append(data)
        self.answer.append(answer)
        self.labels.append(label)

    def extend(self, ids, data, answer,label):
        self.ids.extend(ids)
        self.input.extend(data)
        self.answer.extend(answer)
        self.labels.extend(label)


    def empty(self):
        del (self.ids)
        del (self.input)
        del (self.answer)
        del (self.labels)
        self.ids, self.input, self.answer, self.labels = [], [], [], []

    def __len__(self):
        return len(self.input)

    def __getitem__(self, item):
        return self.ids[item], self.input[item], self.answer[item], self.labels[item]

    def end_at(self, end_index):
        self.ids = self.ids[:end_index]
        self.input = self.input[:end_index]
        self.answer = self.answer[:end_index]
        self.labels = self.labels[:end_index]

    def start_from(self, start_index):
        self.ids = self.ids[start_index:]
        self.input = self.input[start_index:]
        self.answer = self.answer[start_index:]
        self.labels = self.labels[start_index:]




class ModelEvaluator:
    def __init__(self):
        self.start_time = time.time()
        pass

    def print_time(self):
        print(time.time() - self.start_time)

    def get_ceval(self, input_template=None, eos_mark=''):
        '''
        获得CEval评估数据集
        :param input_template:
        :return:
        '''
        data, labels = [], set()
        all_options = []
        dirs = [os.path.join('Tasks', 'Knowledge', 'Benchmarks', 'test', self.task)]
        index = 0
        while (index < len(dirs)):
            dirr = dirs[index]
            for raw_fd in os.listdir(dirr):
                fd = os.path.join(dirr, raw_fd)
                if fd.endswith('csv'):
                    csv_file = pandas.read_csv(fd)
                    for csvline in csv_file.values:
                        question = csvline[1]
                        options = [opt for opt in csvline[2:6]]
                        all_options.extend(options)
                        answer = csvline[6]
                        answer = {'a': 0, 'b': 1, 'c': 2, 'd': 3}.get(answer.lower(), -1)
                        if answer < 0: continue
                        label = str(raw_fd).split('.csv')[0]
                        if '_test' in label: label = label.replace('_test', '')
                        if '_val' in label: label = label.replace('_val', '')
                        data.append((len(data), question, options, answer, [label]))
                        labels.add(label)

                elif os.path.isdir(fd):
                    dirs.append(fd)
            index += 1
        all_options = list(set(all_options))
        for index, (idx, question, options, answer, labels) in enumerate(data):
            while (len(options) < self.options_num):
                opt = random.sample(all_options, k=1)[0]
                if opt in options: continue
                options.append(opt)
            data[index] = (idx, question, options[:self.options_num], answer, labels)

        if self.few_shot > 0:
            label2examples = self.get_traindata(os.path.join('Tasks', 'Knowledge', 'Benchmarks', 'train', 'ceval_train'))
        if self.sample_num > 0: data = random.sample(data, k=min(len(data), self.sample_num))
        infer_dataset = TestDataset()
        if input_template:
            for id, question, options, answer, labels in tqdm.tqdm(data, ncols=80,
                                                                   desc='reading data, construct infer dataset'):
                question = question.replace('[MASK]', '____')

                if few_shot == 0:  # 就正常地进行一个加
                    for opt in options:
                        infer_dataset.append(id,
                                            input_template.format(question=question,
                                                                    options='\n'.join(options),
                                                                    answer=opt,
                                                                    eos=eos_mark),
                                            answer,
                                            '[SEP]'.join(labels))

                else:
                    labels = sorted(labels, key=lambda x: len(x), reverse=True)  # 长的label可能是细粒度的
                    demons = []
                    index = 0
                    while (len(demons) < self.few_shot):
                        mark = True  # 有没有加满self.few_shot个数
                        for label in labels:
                            if index >= len(label2examples[label]): continue  # index超出最长限制
                            ex = label2examples[label][index]
                            if ex in demons: continue  # 已有数据
                            demons.append(ex)
                            mark = False
                            index += 1
                        if mark: break  # 一个example都加不进去了

                    for opt in options:
                        infer_dataset.append(id,
                                            input_template.format(question=question,
                                                                    options='\n'.join(options),
                                                                    answer=opt,
                                                                    demonstrations='\n\n'.join(
                                                                        ['Demonstation %d:\n%s' % (i, j) for i, j in
                                                                         enumerate(demons[:self.few_shot])]),
                                                                    # TODO：考虑下中文prompt
                                                                    eos=eos_mark),
                                             answer,
                                             '[SEP]'.join(labels))

        return infer_dataset

    def get_mmlu(self, input_template=None, eos_mark=''):
        '''
        获得CEval评估数据集
        :param input_template:
        :return:
        '''
        data, labels = [], set()
        all_options = []
        dirs = [os.path.join('Tasks', 'Knowledge', 'Benchmarks','test', self.task)]
        index = 0
        while (index < len(dirs)):
            dirr = dirs[index]
            for raw_fd in os.listdir(dirr):
                fd = os.path.join(dirr, raw_fd)
                if fd.endswith('csv'):
                    csv_file = pandas.read_csv(fd)
                    for csvline in csv_file.values:
                        question = csvline[0]
                        options = [opt.strip() for opt in csvline[1:5]]
                        all_options.extend(options)
                        answer = csvline[5]
                        answer = {'a': 0, 'b': 1, 'c': 2, 'd': 3}.get(answer.lower(), -1)
                        if answer < 0: continue
                        label = str(raw_fd).split('.')[0]
                        if '_test' in label: label = label.replace('_test', '')
                        if '_val' in label: label = label.replace('_val', '')
                        data.append((len(data), question, options, answer, [label]))
                        labels.add(label)
                elif os.path.isdir(fd):
                    dirs.append(fd)
            index += 1
        all_options = list(set(all_options))
        for index, (idx, question, options, answer, labels) in enumerate(data):
            while (len(options) < self.options_num):
                opt = random.sample(all_options, k=1)[0]
                if opt in options: continue
                options.append(opt)
            data[index] = (idx, question, options[:self.options_num], answer, labels)

        if self.few_shot > 0:
            label2examples = self.get_traindata(os.path.join('Tasks', 'Knowledge', 'Benchmarks', 'train', 'mmlu_train'))
        if self.sample_num > 0: data = random.sample(data, k=min(len(data), self.sample_num))
        infer_dataset = TestDataset()
        if input_template:
            for id, question, options, answer, labels in tqdm.tqdm(data, ncols=80,
                                                                   desc='reading data, construct infer dataset'):
                question = question.replace('[MASK]', '____')

                if few_shot == 0:  # 就正常地进行一个加
                    for opt in options:
                        infer_dataset.append(id,
                                            input_template.format(question=question,
                                                                    options='\n'.join(options),
                                                                    answer=opt,
                                                                    eos=eos_mark),
                                            answer,
                                            '[SEP]'.join(labels))

                else:
                    demons = []
                    index = 0
                    while (len(demons) < self.few_shot):
                        mark = True  # 有没有加满self.few_shot个数
                        for label in labels:
                            if index >= len(label2examples[label]): continue  # index超出最长限制
                            ex = label2examples[label][index]
                            if ex in demons: continue  # 已有数据
                            demons.append(ex)
                            mark = False
                            index += 1
                        if mark: break  # 一个example都加不进去了

                    for opt in options:
                        infer_dataset.append(id,
                                            input_template.format(question=question,
                                                                    options='\n'.join(options),
                                                                    answer=opt,
                                                                    demonstrations='\n\n'.join(
                                                                        ['Demonstration %d:\n%s' % (i, j) for i, j in
                                                                         enumerate(demons[:self.few_shot])]),
                                                                    eos=eos_mark),
                                            answer,
                                            '[SEP]'.join(labels))
        return infer_dataset

    def get_traindata(self, dirr):
        label2examples = defaultdict(list)
        if 'xiezhi' in dirr:
            for file in os.listdir(dirr):
                file = os.path.join(dirr, file)
                jsonfile = open(file, encoding='utf-8')
                for line in jsonfile:
                    jsonline = json.loads(line)
                    labels = jsonline['labels']
                    question = jsonline['question']
                    options = jsonline['options'].split('\n')
                    answer = jsonline['answer']
                    if 'eng' in dirr:
                        example = '### question description:\n{question}\n\n### all options:\n{options}\n\n### answer:\n{answer}'
                    else:
                        example = '### 问题描述:\n{question}\n\n### 所有选项:\n{options}\n\n### 答案:\n{answer}'
                    example = example.format(question=question, options='\n'.join(options), answer=answer)

                    for label in labels:
                        label2examples[label].append(example)
        elif 'mmlu' in dirr:
            for file in os.listdir(dirr):
                label = file.split('.csv')[0]
                if '_dev' in label:
                    label = label.replace('_dev', '')
                file = os.path.join(dirr, file)
                csv_file = pandas.read_csv(file)
                for csvline in csv_file.values:
                    question = csvline[0]
                    options = [opt.strip().strip('"').strip("'") for opt in csvline[1:5]]
                    answer = csvline[5]
                    answer = {'a': 0, 'b': 1, 'c': 2, 'd': 3}.get(answer.lower(), -1)

                    example = '### question description:\n{question}\n\n### all options:\n{options}\n\n### answer:\n{answer}'
                    example = example.format(question=question, options='\n'.join(options), answer=options[answer])

                    label2examples[label].append(example)
        elif 'ceval' in dirr:
            for file in os.listdir(dirr):
                label = file.split('.csv')[0]
                if '_dev' in label:
                    label = label.replace('_dev', '')
                file = os.path.join(dirr, file)
                csv_file = pandas.read_csv(file)
                for csvline in csv_file.values:
                    question = csvline[1]
                    options = [opt.strip().strip('"').strip("'") for opt in csvline[2:6]]
                    answer = csvline[6]
                    answer = {'a': 0, 'b': 1, 'c': 2, 'd': 3}.get(answer.lower(), -1)
                    example = '### 问题描述:\n{question}\n\n### 所有选项:\n{options}\n\n### 答案:\n{answer}'
                    example = example.format(question=question, options='\n'.join(options), answer=options[answer])

                    label2examples[label].append(example)
        return label2examples

    def get_xiezhi(self, input_template=None, eos_mark=''):
        '''
        获得獬豸评估数据集
        :param input_template:
        :return:
        '''
        data, all_labels, all_options = [], [], []
        dirs = [os.path.join('Tasks', 'Knowledge', 'Benchmarks','test', self.task)]
        index = 0
        # load data to variable data, all_labels and all_options
        while (index < len(dirs)):
            dirr = dirs[index]
            for raw_fd in os.listdir(dirr):
                fd = os.path.join(dirr, raw_fd)
                if fd.endswith('json'):
                    jsonfile = open(fd, encoding='utf-8')
                    for jsonline in jsonfile:
                        jsonline = json.loads(jsonline)
                        question = jsonline['question']
                        options = [opt.strip().strip('"').strip("'") for opt in jsonline['options'].split('\n')]
                        all_options.extend(options)
                        if jsonline['answer'].strip().strip('"').strip("'") not in options:
                            continue
                        answer = options.index(jsonline['answer'].strip().strip('"').strip("'"))
                        labels = jsonline['labels']

                        data.append((len(data), question, options, answer, labels))
                        all_labels.extend(labels)
                elif os.path.isdir(fd):
                    dirs.append(fd)
            index += 1
        # add more options to data until #options == options_num
        all_options = list(set(all_options))
        for index, (idx, question, options, answer, labels) in enumerate(data):
            while (len(options) < self.options_num):
                opt = random.sample(all_options, k=1)[0]
                if opt in options: continue
                options.append(opt)
            data[index] = (idx, question, options[:self.options_num], answer, labels)

        if self.few_shot > 0:
            if 'eng' in self.task:
                label2examples = self.get_traindata(os.path.join('Tasks', 'Knowledge', 'Benchmarks', 'train', 'xiezhi_train_eng'))
            else:
                label2examples = self.get_traindata(os.path.join('Tasks', 'Knowledge', 'Benchmarks', 'train', 'xiezhi_train_chn'))

        # if random_seed is fixed, the id of answer will be the same to the question id
        if self.sample_num > 0: data = random.sample(data, k=min(len(data), self.sample_num))
        infer_dataset = TestDataset()
        if input_template:
            for id, question, options, answer, labels in tqdm.tqdm(data, ncols=80,
                                                                   desc='reading data, construct infer dataset'):
                if few_shot == 0:  # 就正常地进行一个加
                    for opt in options:
                        infer_dataset.append(id,
                                              input_template.format(question=question,
                                                                    options='\n'.join(options),
                                                                    answer=opt,
                                                                    eos=eos_mark),
                                              answer,
                                              '[SEP]'.join(labels))

                else:
                    labels = sorted(labels, key=lambda x: len(x), reverse=True)  # 长的label可能是细粒度的
                    demons = []
                    index = 0
                    while (len(demons) < self.few_shot):
                        mark = True  # 有没有加满self.few_shot个数
                        for label in labels:
                            if index >= len(label2examples[label]): continue  # index超出最长限制
                            ex = label2examples[label][index]
                            if ex in demons: continue  # 已有数据
                            demons.append(ex)
                            mark = False
                            index += 1
                        if mark: break  # 一个example都加不进去了
                    for opt in options:
                        infer_dataset.append(id,
                                              input_template.format(question=question,
                                                                    options='\n'.join(options),
                                                                    answer=opt,
                                                                    demonstrations='\n\n'.join(
                                                                        ['Demonstation %d:\n%s' % (i, j) for i, j in
                                                                         enumerate(demons[:self.few_shot])]),
                                                                    eos=eos_mark),
                                              answer,
                                              '[SEP]'.join(labels)
                                              )
        return infer_dataset

    def evaluate(self, task, model_name, model_cache_dir, lora_name, lora_cache_dir,
                 result_path='./check_result.json',
                 batch_size=1,
                 sample_num=-1,
                 input_template='### 问题描述:\n{question}\n\n### 所有选项:\n{options}\n\n### 答案:\n{answer}',
                 random_seed=42,
                 few_shot=0, options_num=4, temperature=0.1, topk=40, topp=1, num_beams=1, max_new_tokens=1, model=None,
                 tokenizer=None,
                 **kwargs):
        '''
        :param task:
        :param model_dir:
        :param re_inference:
        :param result_path:
        :param batch_size:
        :param random_seed:
        :param few_shot:
        :param options_num:
        :param temperature:
        :param topk:
        :param topp:
        :param num_beams:
        :return:
        '''
        setup_seed(random_seed)

        self.task = task
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.lora_name = lora_name
        self.lora_cache_dir = lora_cache_dir
        self.batch_size = batch_size
        self.few_shot = few_shot
        self.sample_num = sample_num
        self.temperature = temperature
        self.topk = topk
        self.topp = topp
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.options_num = options_num
        self.result_path = result_path
        self.model = model
        self.tokenizer = tokenizer

        self.generation_config = GenerationConfig(
            temperature=temperature,
            top_p=topp,
            top_k=topk,
            num_beams=num_beams,
            **kwargs
        )
        # load existing log data
        self.start_index = 0  # a index represent one sample, not only one option
        if os.path.exists(self.result_path):
            self.start_index = len(open(self.result_path,encoding='utf-8').readlines())
        else:
            open(self.result_path, 'w', encoding='utf-8')
        if self.start_index < self.sample_num or self.sample_num<0:
            self.fw = open(self.result_path, 'a', encoding='utf-8')
            self._infer(input_template)
            self.fw.close()
        return self.result_path

    def _infer(self, input_template: str, load_in_8bit=False):
        model, tokenizer = get_causalLM(self.model_name, self.model_cache_dir, self.lora_name,self.lora_cache_dir,load_in_8bit)
        self.model_num_params = sum(p.numel() for p in mm.parameters() if p.requires_grad)
        eos_mark = tokenizer.decode(tokenizer.eos_token_id)

        # load benchmark
        datasets = self._get_data(input_template, eos_mark)
        if self.sample_num>=0:
            datasets.end_at(self.sample_num*self.options_num)
        datasets.start_from(self.start_index*self.options_num)

        self.print_time()
        print('### Start From %d Sample' % int(self.start_index + 1))
        print('### End At %d Sample' % self.sample_num)
        print('### Samples Number:', len(datasets) // self.options_num)
        dataloader = DataLoader(datasets, shuffle=False, batch_size=self.batch_size)
        temp_fw = open('temp_file.txt','w',encoding='utf-8')
        criterion = nn.CrossEntropyLoss(reduction='none')
        save_sample = None
        with torch.no_grad():
            for ids, inputs, answers, labels in tqdm.tqdm(dataloader, ncols=80, desc='### Infering Task: %s In %d-shot Setting:' % (self.task, self.few_shot)):
                for input, answer, label in zip(inputs, answers, labels):
                    temp_fw.write('\n\nInput:\n'+'\n'+str(input)+'\n'+'\n\nAnswer:\n'+'\n'+str(answer.item())+'\n'+'\n\nLabel:\n'+'\n'+str(label)+'\n'+'==='*30+'\n'+'==='*30)
                    # print('\n\nInput:\n')
                    # print(input)
                    # print('\n\nAnswer:\n')
                    # print(answer.item())
                    # print('\n\nLabel:\n')
                    # print(label)
                    # print('==='*30)
                    # print('==='*30)
                tokenized_result = tokenizer(inputs, return_tensors='pt', padding=True)
                input_ids = tokenized_result['input_ids'].to('cuda')

                outputs = model.forward(input_ids=input_ids)
                loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), input_ids.view(-1))
                scores = torch.mean(loss.reshape(input_ids.shape), dim=0).detach().cpu().numpy().tolist()

                # 输出每满一个 self.options_num 则保存一下
                for id, score, answer, label in zip(ids, scores, answers, labels):
                    if not save_sample or id != save_sample['line_index']:
                        if save_sample and save_sample['options']:
                            save_sample['model_parameters'] = self.model_num_params
                            self.fw.write(json.dumps(save_sample, ensure_ascii=False) + '\n')
                        save_sample = {'line_index': int(id),
                                       'options': [],
                                       'labels': label.split('[SEP]'),
                                       'answer': int(str(answer).strip('tensor()'))
                                       }
                    save_sample['options'].append(-score)
        if save_sample and save_sample['options']:
            self.fw.write(json.dumps(save_sample, ensure_ascii=False) + '\n')
        # 清空显存
        torch.cuda.empty_cache()

    def _get_data(self, input_template=None, eos_mark=''):
        '''
        获得评估数据集，返回一个数据dataset，一个answer+label dataset
        :param input_template:
        :return:
        '''
        if self.task == 'ceval':
            return self.get_ceval(input_template=input_template, eos_mark=eos_mark)
        elif self.task == 'mmlu':
            return self.get_mmlu(input_template=input_template, eos_mark=eos_mark)
        elif self.task.startswith('xiezhi'):
            return self.get_xiezhi(input_template=input_template, eos_mark=eos_mark)



if __name__ == '__main__':


    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t',
                        default=
                        ['ceval', 'mmlu', 'xiezhi_inter_chn', 'xiezhi_spec_chn', 'xiezhi_inter_eng',
                         'xiezhi_spec_eng'][1])
    parser.add_argument('--model_name', '-mn', default='THUDM/chatglm3-6b')
    parser.add_argument('--model_cache_dir', '-mc', default='../CACHE_DIR')
    parser.add_argument('--lora_name', '-ln', default=None)
    parser.add_argument('--lora_cache_dir', '-lc', default='../CACHE_DIR')
    parser.add_argument('--sample_num', '-sn', default=-1, type=int)
    parser.add_argument('--language', '-lang', default='eng', type=str)
    parser.add_argument('--batch_size', '-bs', default=4, type=int)
    parser.add_argument('--options_num', '-on', default=8, type=int)
    parser.add_argument('--need_inference', '-infer', default=True, type=bool)
    parser.add_argument('--result_path', '-path', default='./Knowledge_Evaluator/output_results/check_result')
    parser.add_argument('--metric', '-m', default='hit1')
    parser.add_argument('--random_seed', '-rs', default=42, type=int)
    parser.add_argument('--few_shot', '-fs', default=2, type=int)
    parser.add_argument('--temperature', '-tmp', default=0.1, type=float)
    parser.add_argument('--topk', '-pk', default=40, type=int)
    parser.add_argument('--topp', '-pp', default=1, type=float)
    parser.add_argument('--num_beams', '-nb', default=2, type=int)
    parser.add_argument('--max_new_tokens', '-mt', default=1, type=int)

    args = parser.parse_args()

    task = args.task

    model_name = args.model_name
    model_cache_dir = args.model_cache_dir
    lora_name = args.lora_name
    lora_cache_dir = args.lora_cache_dir
    sample_num = args.sample_num
    language = args.language
    batch_size = args.batch_size
    options_num = args.options_num
    need_inference = args.need_inference
    result_path = args.result_path
    metric = args.metric
    random_seed = args.random_seed
    few_shot = args.few_shot
    temperature = args.temperature
    topk = args.topk
    topp = args.topp
    num_beams = args.num_beams
    max_new_tokens = args.max_new_tokens

    UID = '%s_%s_%s_%s_%s_%s' % (model_name, lora_name, task, few_shot, options_num, random_seed)
    UID = UID.replace('/', '').replace('\\', '')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    check_result_path = os.path.join(result_path, UID + '.json')
    print('### You\'re using task: %s' % task)
    print('### Mission UID: %s' % UID)
    if need_inference:

        # prepare prompts
        cn_multiple_choice_template = '下面请你作为一个答题者，回答一道选择题。我会在“问题描述”中描述这道问题是什么、“所有选项”中列出所有可以选择的选项、请你将你认为正确的选项输出在“答案”后面。请你将你认为正确的选项内容进行输出，不要输出多余的内容。\n\n### 问题描述:\n{question}\n\n### 所有选项:\n{options}\n\n### 答案:\n{answer} {eos}'
        en_multiple_choice_template = 'Below you will be asked to answer a multiple choice question as a respondent. I will describe what the question is in the "Question Description", list all available options in the "All Options", and ask you to output the option you think is correct after the "Answer" field. Please do not output any extra content, only output the text of the option which you think is right. \n\n### question description:\n{question}\n\n### all options:\n{options}\n\n### answer:\n{answer} {eos}'
        cn_multiple_choice_demonstration_template = '下面请你作为一个答题者，回答一道选择题。我会在“问题描述”中描述这道问题是什么、“所有选项”中列出所有可以选择的选项、请你将你认为正确的选项输出在“答案”后面。我会在前面给你几个例子，并用“Demonstration”符号进行标记，请你根据给出的例子的格式和可以借鉴的信息，只回答最后一题。请你将你认为正确的选项内容进行输出，不要输出多余的文本。\n\n{demonstrations}\n\nQuestion:\n### 问题描述:\n{question}\n\n### 所有选项:\n{options}\n\n### 答案:\n{answer} {eos}'
        en_multiple_choice_demonstration_template = 'Below you will be asked to answer a multiple choice question as a respondent. I will describe what the question is in the "Question Description", list all available options in the "All Options", and ask you to output the option you think is correct after the "Answer" field. I will give you a few examples up front, marked with the "Demonstration" word. You only need to answer the last question. Please follow the format of the examples given and draw on all the possible information in the example to raise the chance to correctly answer the question. Please do not output any extra content, only output the text of the option which you think is right. \n\n{demonstrations}\n\nQuestion:\n### question description:\n{question}\n\n### all options:\n{options}\n\n### answer:\n{answer} {eos}'
        if few_shot > 0 and language == 'chn':  # 只有xiezhi提供few shot demonstration
            input_template = cn_multiple_choice_demonstration_template
        elif few_shot > 0 and language == 'eng':  # 只有xiezhi提供few shot demonstration
            input_template = en_multiple_choice_demonstration_template
        elif language == 'chn':
            input_template = cn_multiple_choice_template
        elif language == 'eng':
            input_template = en_multiple_choice_template
        else:
            input_template = en_multiple_choice_template  # 其他的待定

        # Prepare Evaluator
        Evaluator = ModelEvaluator()
        # Start Evaluation
        check_result_path = Evaluator.evaluate(task, model_name, model_cache_dir, lora_name, lora_cache_dir,
                        check_result_path, batch_size, sample_num,
                        input_template, random_seed, few_shot,
                        options_num, temperature, topk, topp, num_beams, max_new_tokens)

    # from conclude import *
    #
    # label2result = get_evaluation_result_from_check_path(check_result_path)
    # label2hitk = get_hitk_result(label2result, int(metric.strip('hit')))
    # label2mrr = get_mrr_result(label2result)
    #
    # for label in label2hitk:
    #     if label2mrr[label]['num'] < 200: continue
    #     print('Label: %s, Hit@4: %.3f, MRR: %.3f, Num: %d'%(label, label2hitk[label]['mean'], label2mrr[label]['mean'], label2mrr[label]['num']))



