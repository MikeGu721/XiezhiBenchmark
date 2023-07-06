python model_test.py \
  --sample_num=1000 \
  --options_num=8 \
  --few_shot=3 \
  --task=xiezhi_inter \
  --metric=mrr
  --model_cache_dir=../../LLM/CACHE_DIR


#  --model_name'  模型名称
#  --model_cache_dir'  模型存放位置
#  --lora_name'  lora名称
#  --lora_cache_dir'  lora存放位置
#  --sample_num'  评估样本个数
#  --language'  语种
#  --batch_size'
#  --options_num'  设置选项个数
#  --re_inference'  True = 直接读取现有结果不再重跑代码
#  --result_path'  结果存放位置
#  --metric'  度量指标，支持“mrr“和”hitn“，n为小于options_num的数字
#  --random_seed'
#  --few_shot'  样例个数
#  --temperature'
#  --topk'
#  --topp'
#  --num_beams'
#  --max_new_tokens'  在xiezhi论文的设定中，max_new_tokens为1
