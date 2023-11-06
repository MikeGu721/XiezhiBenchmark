#python evaluate.py \
#  --sample_num=1000 \
#  --options_num=8 \
#  --language=chn \
#  --model_name=THUDM/chatglm3-6b \
#  --few_shot=3 \
#  --task=xiezhi_inter_eng \
#  --metric=mrr
#  --model_cache_dir=../../CACHE_DIR
#
#read -n 1

options_nums=(4 8 16 32 50)
few_shots=(0 1 2 3)
glm_ver=(chatglm-6b chatglm2-6b chatglm3-6b)
task=(ceval mmlu)
# 两重循环
for task in ${task[@]}
do
  for glm_ver in ${glm_ver[@]}
  do
    for options_num in ${options_nums[@]}
    do
      for few_shot in ${few_shots[@]}
      do
            # 执行你的代码
        python evaluate.py \
          --sample_num=-1 \
          --options_num=$options_num \
          --language=chn \
          --task=$task \
          --model_name=THUDM/$glm_ver \
          --few_shot=$few_shot \
          --task=xiezhi_inter_eng \
          --metric=mrr \
          --model_cache_dir=../../CACHE_DIR
      done
    done
  done
done


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
