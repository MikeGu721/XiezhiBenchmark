import os
from tqdm import tqdm
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
from accelerate import init_empty_weights,load_checkpoint_and_dispatch
from evaluators.evaluator import Evaluator
from time import sleep
import re


class Moss_Evaluator(Evaluator):

    def __init__(self,choices,k,model_name):
        super(Moss_Evaluator,self).__init__(choices,model_name,k)

        # download the model from huggingface hub
        model_path="fnlp/moss-moon-003-sft"
        if not os.path.exists(model_path):
            model_path=snapshot_download(model_path)

        # load the model
        self.config=AutoConfig.from_pretrained("fnlp/moss-moon-003-sft",trust_remote_code=True)
        self.tokenizer=AutoTokenizer.from_pretrained("fnlp/moss-moon-003-sft",trust_remote_code=True)
        self.tokenizer.padding_side="left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        with init_empty_weights():
            self.model=AutoModelForCausalLM.from_config(self.config,torch_dtype=torch.float16,trust_remote_code=True)
        self.model.tie_weights()
        self.model=load_checkpoint_and_dispatch(self.model,model_path,device_map="auto",no_split_module_classes=["MossBlock"],dtype=torch.float16,offload_folder="/home/parallels/MOSS/temp",)

    # format one question for demostration
    def format_example(self,line,include_answer=True,cot=False):
        example=line['question']
        for choice in self.choices:
            example+=f'\n{choice}. {line[f"{choice}"]}'

        example+='\n答案：'
        if include_answer:
            if cot:
                ans=line["answer"]
                content="让我们一步一步思考，\n"+line["explanation"]+f"\n所以答案是{ans}。"
                return [
                    {"role":"user","content":example},
                    {"role":"assistant","content":content}
                ]
            else:
                return [
                    {"role":"user","content":example},
                    {"role":"assistant","content":line["answer"]}
                ]
        else:
            return [
                {"role":"user","content":example},
            ]

    # using format_example() to generate 
    def generate_few_shot_prompt(self,subject,dev_df,cot=False):
        prompt=f"你是一个中文人工智能助手，以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n"
        k=self.k
        if self.k==-1:
            k=dev_df.shape[0]
        for i in range(k):
            tmp=self.format_example(dev_df.iloc[i,:],include_answer=True,cot=cot)
            if i==0:
                tmp[0]["content"]=f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n"+tmp[0]["content"]
            user=tmp[0]['content']
            moss=tmp[1]['content']
            prompt+=f"<|Human|>: {user} <eoh>\n<|MOSS|>: {moss} <eom>\n"
        return prompt

    # main function of evaluation
    def eval_subject(self,subject_name,test_df,dev_df=None,few_shot=False,save_result_dir=None,cot=False):
        # preparation
        correct_num=0
        if save_result_dir:
            result=[]
            score=[]
        if few_shot:
            few_shot_prompt=self.generate_few_shot_prompt(subject_name,dev_df,cot=cot)
        else:
            few_shot_prompt=f"你是一个中文人工智能助手，以下是中国关于{subject_name}考试的单项选择题，请选出其中的正确答案。\n"
        answers=list(test_df['answer'])
        message_list=[]
        tar_list=[]

        # evaluation
        for row_index,row in tqdm(test_df.iterrows(),total=len(test_df)):
            # prepare prompt and answer
            question=self.format_example(row,include_answer=False)
            full_prompt=few_shot_prompt+"<|Human|>: "+question[0]['content']+" <eoh>\n<|MOSS|>:"
            message_list.append(full_prompt)
            tar_list.append(answers[row_index])


            if len(message_list)%1==0 or row_index==len(test_df)-1:
                # 使用tokenizer对输入的消息集合进行编码，变成模型接受的张量形式，这里参数中return_tensors="pt"表示返回pytorch张量格式，padding=True表示填充
                inputs=self.tokenizer(message_list,return_tensors="pt",padding=True)
                for k in inputs:
                    # 将input张量放在GPU上执行
                    inputs[k]=inputs[k].cuda()
                # 计算最大能够生成的令牌数，这里假定每个回复中最多允许生成2040个令牌，然后减去输入的已有令牌数
                max_tok=2040-inputs.input_ids.shape[1]
                # 使用预训练语言模型根据输入生成回复。函数generate()会返回生成的文本序列，参数do_sample=True表示使用采样方式生成文本、temperature表示温度参数，值越小生成的词越保守、top_p表示需要满足的最小条件概率总量、 repetition_penalty表示控制重复生成的惩罚因子、max_new_tokens指生成文本的最大长度
                outputs=self.model.generate(**inputs,do_sample=True,temperature=0.2,top_p=0.8,repetition_penalty=1.02,
                                            max_new_tokens=max_tok)
                # 计算输入的文本在编码后的长度，输入文本的编码有一个重要部分即是mask，此时取这个mask的和，并按句子长度取max，即可得到这里的input_len，用于指示输入序列需要从哪里开始解码
                input_len=torch.max(torch.sum(inputs.attention_mask,axis=1))
                # 将模型生成的回复从张量类型变为字符串类型，忽略掉回复中的特殊字符(mask_id、unk_id、[CLS], [SEP]等)，存储在response_list中，并输出给用户
                response_list=[
                    self.tokenizer.decode(outputs[i][input_len:],skip_special_tokens=True)
                    for i in range(outputs.shape[0])
                ]

                # match the answer and calculate the accuracy
                for i,response_str in enumerate(response_list):
                    #print(response_str)
                    if cot:
                        ans_list=re.findall(r"答案是(.+?)。",response_str)
                        if len(ans_list)==0:
                            ans_list=re.findall(r"答案为(.+?)。",response_str)
                        if len(ans_list)==0:
                            ans_list=re.findall(r"选项(.+?)是正确的。",response_str)
                        if len(ans_list)==0:
                            ans_list=re.findall(r"选择选项(.+?)",response_str)
                        if len(ans_list)==0:
                            correct=0
                        else:
                            if self.exact_match(ans_list[-1],tar_list[i]):
                                correct_num+=1
                                correct=1
                            else:
                                correct=0
                    else:
                        response_str=response_str.strip()
                        if few_shot:
                            if self.exact_match(response_str,tar_list[i]):
                                correct_num+=1
                                correct=1
                            else:
                                correct=0
                        else:
                            if response_str[0]==tar_list[i]:
                                correct_num+=1
                                correct=1
                            else:
                                correct=0

                    if save_result_dir:
                        result.append(response_str)
                        score.append(correct)
                message_list=[]
                tar_list=[]

        correct_ratio=100*correct_num/len(answers)

        if save_result_dir:
            test_df['model_output']=result
            test_df["correctness"]=score
            test_df.to_csv(os.path.join(save_result_dir,f'{subject_name}_val.csv'),encoding="utf-8",index=False)
        return correct_ratio
