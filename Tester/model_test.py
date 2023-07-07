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


def get_causalLM(model_name, model_cache_dir, lora_name, lora_cache_dir, load_in_8bit=False):
    causalLM = \
        LlamaForCausalLM if 'llama' in model_name.lower() or 'cute' in model_name.lower() else \
            BloomForCausalLM if 'bloom' in model_name.lower() or 'belle' in model_name.lower() else \
                AutoModel if 'glm' in model_name.lower() else \
                    GPTNeoXForCausalLM if 'pythia' in model_name.lower() else \
                        AutoModelForCausalLM

    device_map = 'balanced_low_0'
    # device_map = 'auto'
    if model_cache_dir:
        model = causalLM.from_pretrained(model_name, cache_dir=model_cache_dir,
                                         torch_dtype=torch.float16, device_map='auto',
                                         load_in_8bit=load_in_8bit, trust_remote_code=True)
    else:
        model = causalLM.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir='../LLM/CACHE_DIR',
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
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir='../LLM/CACHE_DIR')
    else:
        tokenizer = LlamaTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir='../LLM/CACHE_DIR')
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
    def __init__(self, ids=None, data=None):
        super().__init__()
        self.ids, self.input = [], []
        if ids:
            self.extend(ids, data)

    def append(self, id, data):
        self.ids.append(id)
        self.input.append(data)

    def extend(self, ids, data):
        self.ids.extend(ids)
        self.input.extend(data)

    def empty(self):
        del (self.ids)
        del (self.input)
        self.ids, self.input = [], []

    def __len__(self):
        return len(self.input)

    def __getitem__(self, item):
        return self.ids[item], self.input[item]

    def start_from(self, start_index):
        self.ids = self.ids[start_index:]  # 此处start_index要除去只有\n的这种行
        self.input = self.input[start_index:]


def hitk(pred, result, k):
    try:
        pred_arg = list(np.argsort(pred))[::-1]
        if result in pred_arg[:k]:
            return 1
        else:
            return 0
    except:
        return 0


def mrr(pred, result):
    try:
        pred_arg = list(np.argsort(pred)[::-1])
        return 1 / (pred_arg.index(result) + 1)
    except:
        return 0


class ModelTester:
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
        dirs = [os.path.join('tasks', self.task)]
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
            label2examples = self.get_traindata(os.path.join('tasks', 'ceval_train'))
        answer_datasets = TestDataset()
        for id, question, options, answer, label in tqdm.tqdm(data, ncols=80,
                                                              desc='reading data, construct answer dataset'):
            answer_datasets.append(id, (answer, label))
        if self.sample_num > 0: data = random.sample(data, k=min(len(data), self.sample_num))
        infer_datasets = TestDataset()
        if input_template:
            for id, question, options, answer, labels in tqdm.tqdm(data, ncols=80,
                                                                   desc='reading data, construct infer dataset'):
                question = question.replace('[MASK]', '____')

                if few_shot == 0:  # 就正常地进行一个加
                    for opt in options:
                        infer_datasets.append(id,
                                              input_template.format(question=question,
                                                                    options='\n'.join(options),
                                                                    answer=opt,
                                                                    eos=eos_mark))

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
                        infer_datasets.append(id,
                                              input_template.format(question=question,
                                                                    options='\n'.join(options),
                                                                    answer=opt,
                                                                    demonstrations='\n\n'.join(
                                                                        ['Demonstation %d:\n%s' % (i, j) for i, j in
                                                                         enumerate(demons[:self.few_shot])]),
                                                                    # TODO：考虑下中文prompt
                                                                    eos=eos_mark))
        return infer_datasets, answer_datasets, labels

    def get_mmlu(self, input_template=None, eos_mark=''):
        '''
        获得CEval评估数据集
        :param input_template:
        :return:
        '''
        data, labels = [], set()
        all_options = []
        dirs = [os.path.join('tasks', self.task)]
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
            label2examples = self.get_traindata(os.path.join('tasks', 'mmlu_train'))
        answer_datasets = TestDataset()
        for id, question, options, answer, label in tqdm.tqdm(data, ncols=80,
                                                              desc='reading data, construct answer dataset'):
            answer_datasets.append(id, (answer, label))
        if self.sample_num > 0: data = random.sample(data, k=min(len(data), self.sample_num))
        infer_datasets = TestDataset()
        if input_template:
            for id, question, options, answer, labels in tqdm.tqdm(data, ncols=80,
                                                                   desc='reading data, construct infer dataset'):
                question = question.replace('[MASK]', '____')

                if few_shot == 0:  # 就正常地进行一个加
                    for opt in options:
                        infer_datasets.append(id,
                                              input_template.format(question=question,
                                                                    options='\n'.join(options),
                                                                    answer=opt,
                                                                    eos=eos_mark))

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
                        infer_datasets.append(id,
                                              input_template.format(question=question,
                                                                    options='\n'.join(options),
                                                                    answer=opt,
                                                                    demonstrations='\n\n'.join(
                                                                        ['Demonstration %d:\n%s' % (i, j) for i, j in
                                                                         enumerate(demons[:self.few_shot])]),
                                                                    # TODO：考虑下中文prompt
                                                                    eos=eos_mark))
        return infer_datasets, answer_datasets, labels

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
                    options = jsonline['options']
                    answer = jsonline['answer']
                    if 'eng' in dirr:
                        example = '### question description:\n```{question}```\n\n### all options:\n```{options}```\n\n### answer:\n```{answer}```'
                    else:
                        example = '### 问题描述:\n```{question}```\n\n### 所有选项:\n```{options}```\n\n### 答案:\n```{answer}```'
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

                    example = '### question description:\n```{question}```\n\n### all options:\n```{options}```\n\n### answer:\n```{answer}```'
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
                    example = '### 问题描述:\n```{question}```\n\n### 所有选项:\n```{options}```\n\n### 答案:\n```{answer}```'
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
        dirs = [os.path.join('tasks', self.task)]
        index = 0
        while (index < len(dirs)):
            dirr = dirs[index]
            for raw_fd in os.listdir(dirr):
                fd = os.path.join(dirr, raw_fd)
                if fd.endswith('json'):
                    jsonfile = open(fd, encoding='utf-8')
                    for jsonline in jsonfile:
                        jsonline = json.loads(jsonline)

                        question = jsonline['question']
                        options = [opt.strip().strip('"').strip("'") for opt in jsonline['options']]
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
        all_options = list(set(all_options))
        for index, (idx, question, options, answer, labels) in enumerate(data):
            while (len(options) < self.options_num):
                opt = random.sample(all_options, k=1)[0]
                if opt in options: continue
                options.append(opt)
            data[index] = (idx, question, options[:self.options_num], answer, labels)

        if self.few_shot > 0:
            if 'eng' in self.task:
                label2examples = self.get_traindata(os.path.join('tasks', 'xiezhi_train_eng'))
            else:
                label2examples = self.get_traindata(os.path.join('tasks', 'xiezhi_train_chn'))

        answer_datasets = TestDataset()
        for id, question, options, answer, label in tqdm.tqdm(data, ncols=80,
                                                              desc='reading data, construct answer dataset'):
            answer_datasets.append(id, (answer, label))
        if self.sample_num > 0: data = random.sample(data, k=min(len(data), self.sample_num))
        infer_datasets = TestDataset()
        if input_template:
            for id, question, options, answer, labels in tqdm.tqdm(data, ncols=80,
                                                                   desc='reading data, construct infer dataset'):
                if few_shot == 0:  # 就正常地进行一个加
                    for opt in options:
                        infer_datasets.append(id,
                                              input_template.format(question=question,
                                                                    options='\n'.join(options),
                                                                    answer=opt,
                                                                    eos=eos_mark))

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
                        infer_datasets.append(id,
                                              input_template.format(question=question,
                                                                    options='\n'.join(options),
                                                                    answer=opt,
                                                                    demonstrations='\n\n'.join(
                                                                        ['Demonstation %d:\n%s' % (i, j) for i, j in
                                                                         enumerate(demons[:self.few_shot])]),
                                                                    # TODO：考虑下中文prompt
                                                                    eos=eos_mark))
        return infer_datasets, answer_datasets, set(all_labels)

    def get_m3ke(self, input_template=None, eos_mark=''):
        '''
        获得獬豸评估数据集
        :param input_template:
        :return:
        '''
        data, all_labels, all_options = [], [], []
        dirs = [os.path.join('tasks', self.task)]
        index = 0
        while (index < len(dirs)):
            dirr = dirs[index]
            for raw_fd in os.listdir(dirr):
                fd = os.path.join(dirr, raw_fd)
                if fd.endswith('xlsx'):
                    excelfile = pd.read_excel(fd)
                    for value in excelfile.values:
                        question = value[0]
                        options = [str(opt).strip().strip('"').strip("'") for opt in value[1:5]]
                        all_options.extend(options)
                        answer = value[5]
                        answer = {'a': 0, 'b': 1, 'c': 2, 'd': 3}.get(answer.lower(), -1)
                        if answer == -1: continue
                        if answer >= self.options_num: continue
                        labels = [str(raw_fd).split('.xlsx')[0]]

                        data.append((len(data), question, options, answer, labels))
                        all_labels.extend(labels)
                elif os.path.isdir(fd):
                    dirs.append(fd)
            index += 1
        all_options = list(set(all_options))
        for index, (idx, question, options, answer, labels) in enumerate(data):
            while (len(options) < self.options_num):
                opt = random.sample(all_options, k=1)[0]
                if opt in options: continue
                options.append(opt)
            options = options[:self.options_num]
            data[index] = (idx, question, options, answer, labels)

        answer_datasets = TestDataset()
        for id, question, options, answer, label in tqdm.tqdm(data, ncols=80, desc='reading data'):
            answer_datasets.append(id, (answer, label))
        if self.sample_num > 0: data = random.sample(data, k=min(len(data), self.sample_num))
        infer_datasets = TestDataset()
        if input_template:
            for id, question, options, answer, label in tqdm.tqdm(data, ncols=80, desc='reading data'):
                for opt in options:
                    infer_datasets.append(id,
                                          input_template.format(question=question, options='\n'.join(options),
                                                                answer=opt, eos=eos_mark))
        return infer_datasets, answer_datasets, set(all_labels)

    def test(self, task, model_name, model_cache_dir, lora_name, lora_cache_dir,
             re_inference=True, metric='hit1', result_path='./check_result.json', batch_size=1,
             sample_num=-1,
             input_template='### 问题描述:\n{question}\n\n### 所有选项:\n{options}\n\n### 答案:\n{answer}',
             random_seed=42,
             few_shot=0, options_num=4, temperature=0.1, topk=40, topp=1, num_beams=1, max_new_tokens=1, model=None,
             tokenizer=None, **kwargs):
        '''
        :param task:
        :param model_dir:
        :param re_inference:
        :param metric:
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
        assert metric.startswith('hit') or metric in ['acc', 'mrr']
        if metric == 'acc': metric = 'hit1'
        setup_seed(random_seed)

        self.task = task
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.lora_name = lora_name
        self.lora_cache_dir = lora_cache_dir
        self.metric = metric
        self.batch_size = batch_size
        self.few_shot = few_shot
        self.sample_num = sample_num
        self.temperature = temperature
        self.topk = topk
        self.topp = topp
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.options_num = options_num

        self.model = model
        self.tokenizer = tokenizer

        self.generation_config = GenerationConfig(
            temperature=temperature,
            top_p=topp,
            top_k=topk,
            num_beams=num_beams,
            **kwargs
        )
        # 任务标识符
        self.UID = '%s_%s_%s_%s_%s_%s' % (
            model_name, lora_name, task, few_shot, options_num, random_seed)
        self.UID = self.UID.replace('/', '').replace('\\', '')
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        self.result_path = os.path.join(result_path, self.UID + '.json')
        print('### You\'re using task: %s' % self.task)
        print('### Mission UID: %s' % self.UID)

        if re_inference:
            # 推理到了从哪条数据开始继续推理
            self.start_index = 0
            if os.path.exists(self.result_path):
                self.start_index = len([i for i in open(self.result_path, encoding='utf-8') if i.strip()])
            else:
                open(self.result_path, 'w', encoding='utf-8')
            self.fw = open(self.result_path, 'a', encoding='utf-8')
            self._infer(input_template)
            self.fw.close()

        if not os.path.exists(self.result_path): return {}, {}
        # 历史遗留问题，需要规整下模型的输出结果
        temp = [json.loads(line) for line in open(self.result_path, encoding='utf-8')]
        id2pred = {}
        for id, score, output in temp:
            if type(id) == str and 'tensor' in id:
                id = id.strip('tensor()')
            if int(id) not in id2pred:
                id2pred[int(id)] = [[], []]
            id2pred[int(id)][0].append(score)
            id2pred[int(id)][1].append(output)
        id2result, label2result = self._verify(id2pred)
        return id2result, label2result

    def _infer(self, input_template: str, load_in_8bit=False):
        if not self.model:
            if 'models--' in self.model_cache_dir:
                model, tokenizer = get_causalLM(self.model_cache_dir, None, self.lora_cache_dir, None, load_in_8bit)
            else:
                try:
                    model, tokenizer = get_causalLM(self.model_name, self.model_cache_dir, self.lora_name,
                                                    self.lora_cache_dir,
                                                    load_in_8bit)
                except:
                    input_model_name, input_lora_name = get_names(model_name, lora_name, self.model_cache_dir)
                    model, tokenizer = get_causalLM(input_model_name, None, input_lora_name, None,
                                                    load_in_8bit)
        else:
            model = self.model
            tokenizer = self.tokenizer
        eos_mark = tokenizer.decode(tokenizer.eos_token_id)
        datasets, _, _ = self._get_data(input_template, eos_mark)
        datasets.start_from(self.start_index)
        self.print_time()
        print('### Start From %d Sample' % self.start_index)
        print('### Samples Number:', len(datasets) // self.options_num)
        dataloader = DataLoader(datasets, shuffle=False, batch_size=self.batch_size)
        with torch.no_grad():
            for ids, inputs in tqdm.tqdm(dataloader, ncols=80,
                                         desc='infering on task: %s in %d-shot setting' % (self.task, self.few_shot)):
                try:
                    input_ids = tokenizer(inputs, return_tensors='pt', padding=True)['input_ids'].to('cuda')
                    generation_output = model.generate(
                        input_ids=input_ids,
                        generation_config=self.generation_config,
                        return_dict_in_generate=True,
                        pad_token_id=model.config.pad_token_id,
                        eos_token_id=model.config.eos_token_id,
                        bos_token_id=model.config.bos_token_id,
                        output_scores=True,
                        max_new_tokens=max_new_tokens)

                    scores = generation_output.sequences_scores
                    scores = scores.detach().cpu().numpy().tolist()
                    ss = generation_output.sequences
                    outputs = [tokenizer.decode(s) for s in ss]
                except:
                    scores = [-1 for i in range(len(ids))]
                    outputs = ['error' for i in range(len(ids))]
                    # 输出模型结果
                for id, score, output in zip(ids, scores, outputs):
                    self.fw.write(json.dumps([int(id), score, output], ensure_ascii=False) + '\n')

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
        elif self.task == 'm3ke':
            return self.get_m3ke(input_template=input_template, eos_mark=eos_mark)

    def _verify(self, id2pred):
        id2result = {}
        label2result = defaultdict(list)
        _, label_dataset, all_labels = self._get_data()
        self.print_time()
        print('### Label Number:', len(all_labels))
        for id, (answer, labels) in tqdm.tqdm(label_dataset, ncols=80, desc='verifying'):
            if type(id) == str and 'tensor' in id:
                id = id.strip('tensor()')
            if int(id) not in id2pred: continue
            if self.metric.startswith('hit'):
                ans = hitk(id2pred[int(id)][0], answer, int(self.metric.split('hit')[-1]))
            elif self.metric.startswith('mrr'):
                ans = mrr(id2pred[id][0], answer)
            id2result[id] = ans
            for label in labels:
                label2result[label].append(ans)
        label2result = {label: {'mean': np.mean(label2result[label]), 'std': np.std(label2result[label]),
                                'num': len(label2result[label])} for label in
                        label2result}
        return id2result, label2result


if __name__ == '__main__':
    cn_multiple_choice_template = '### 问题描述:\n```{question}```\n\n### 所有选项:\n```{options}```\n\n### 答案:\n```{answer}``` {eos}'
    en_multiple_choice_template = '### question description:\n```{question}```\n\n### all options:\n```{options}```\n\n### answer:\n```{answer}``` {eos}'
    cn_multiple_choice_demonstration_template = '\n```{demonstrations}```\n\n### 问题描述:\n```{question}```\n\n### 所有选项:\n```{options}```\n\n### 答案:\n```{answer}``` {eos}'
    en_multiple_choice_demonstration_template = '```{demonstrations}```\n\n### question description:\n```{question}```\n\n### all options:\n```{options}```### answer:\n```{answer}``` {eos}'

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t',
                        default=
                        ['ceval', 'mmlu', 'm3ke', 'xiezhi_inter_chn', 'xiezhi_spec_chn', 'xiezhi_inter_eng', 'xiezhi_spec_eng'][
                            4])
    parser.add_argument('--model_name', '-mn', default='bigscience/bloomz-7b1')
    parser.add_argument('--model_cache_dir', '-mc', default='../LLM/CACHE_DIR')
    parser.add_argument('--lora_name', '-ln', default=None)
    parser.add_argument('--lora_cache_dir', '-lc', default='../LLM/CACHE_DIR')
    parser.add_argument('--sample_num', '-sn', default=1000, type=int)
    parser.add_argument('--language', '-lang', default='chn', type=str)
    parser.add_argument('--batch_size', '-bs', default=8, type=int)
    parser.add_argument('--options_num', '-on', default=4, type=int)
    parser.add_argument('--re_inference', '-infer', default=True, type=bool)
    parser.add_argument('--result_path', '-path', default='./output_results/check_result')
    parser.add_argument('--metric', '-m', default='hit1')
    parser.add_argument('--random_seed', '-rs', default=42, type=int)
    parser.add_argument('--few_shot', '-fs', default=0, type=int)
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
    re_inference = args.re_inference
    result_path = args.result_path
    metric = args.metric
    random_seed = args.random_seed
    few_shot = args.few_shot
    temperature = args.temperature
    topk = args.topk
    topp = args.topp
    num_beams = args.num_beams
    max_new_tokens = args.max_new_tokens

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

    Tester = ModelTester()
    id2result, label2result = Tester.test(task, model_name, model_cache_dir, lora_name, lora_cache_dir,
                                          re_inference, metric, result_path, batch_size, sample_num,
                                          input_template, random_seed, few_shot,
                                          options_num, temperature, topk, topp, num_beams, max_new_tokens)

    output = {}
    for label in label2result:
        output[label] = label2result.get(label, 'NONE')
    output['OVERALL'] = {'mean': np.mean([i[1] for i in id2result.items()]),
                         'std': np.std([i[1] for i in id2result.items()])}
    [print(label, output[label]) for label in output]
    open('./results/%s_result.json' % task, 'w', encoding='utf-8').write(
        json.dumps(output, ensure_ascii=False, indent=4))
