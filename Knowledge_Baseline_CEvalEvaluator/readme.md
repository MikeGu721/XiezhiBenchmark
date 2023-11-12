# README

## Moss, ChatGLM and MiniMax

* Run the evaluation script `code/evaluator_series/eval.py` from the project root directory with the following optional arguments:

  ```
  --openai_key specifies the OpenAI key
  --minimax_group_id and --minimax_key specifies the group-id and api-key (for MiniMax)
  --task specifies task to be tested
  --cuda_device specifies the cuda device to be used (for ChatGLM)
  --sample_num specifies the sample number to be evaluated
  ```

Usage examples:

Please find examples in model_test_by_ceval.sh

## Information for developer 

The following lists the interfaces for the core part of the Q&A interactions in the use of each model.

### 1 minimax
Core call: send it with a POST request, get a returned JSON, take the reply from it.

```python
self.url = f"https://api.minimax.chat/v1/text/chatcompletion?GroupId={self.group_id}"
self.headers = {
    "Authorization": f"Bearer {self.api_key}",
    "Content-Type": "application/json"
}
response = requests.request("POST", self.url, headers=self.headers, json=data).json()
if response['base_resp']['status_msg'] == 'success':
    return response['reply'].strip()
```

Related Documents：https://api.minimax.chat/document/guides/chat?id=6433f37294878d408fc82953

### 2 chatgpt
Core call：call create funtion，return a dist.
```python
response = openai.ChatCompletion.create(
                        model=self.model_name,
                        messages=full_prompt,
                        temperature=0.
                    )
response_str = response['choices'][0]['message']['content']
```

Related Documents：https://platform.openai.com/docs/api-reference/chat/create?lang=python

### 3 moss
```python
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
```

Related Documents：https://github.com/OpenLMLab/MOSS#%E6%A8%A1%E5%9E%8B

### 4 chatglm
```python
question = self.format_example(row, include_answer=False, cot=cot)
if few_shot:
    response, _ = self.model.chat(self.tokenizer, question, do_sample=False, history=history)
    response = response.strip()
    # For ChatGLM, we use answer extraction in answer-only mode too.
    ans, direct_extract = self.extract_cot_answer(row, response)
else:   
    # zero-shot by extracting answer from distribution
    # use generate_dist() to extract
    ans = self.generate_dist(self.model, self.tokenizer, question, do_sample=False, max_length=2048, history=history)
```

Related Documents：https://github.com/THUDM/ChatGLM-6B

Among them, the generate_dist function is more complex, internally calling the generate function of class transformers.GenerationMixin directly to generate the text. generate function's parameters include the predefined GenerationConfig family of parameters.

Related Documents：
https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/text_generation#transformers.GenerationMixin.generate
https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/text_generation#transformers.GenerationConfig
