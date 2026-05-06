./kg_retrieval_launch_cwq.sh

bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn7_hf.sh JinyeopSong/KG-R1_test cwq \
  reward_model.reward_kwargs.use_legacy_entity_f1=true \
  reward_model.reward_kwargs.use_exact_match_binary_for_passk=false

bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn5_hf.sh JinyeopSong/KG-R1_test cwq \
  reward_model.reward_kwargs.use_legacy_entity_f1=false \
  reward_model.reward_kwargs.use_exact_match_binary_for_passk=true
bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn7_hf.sh JinyeopSong/KG-R1_test cwq \
  reward_model.reward_kwargs.use_legacy_entity_f1=false \
  reward_model.reward_kwargs.use_exact_match_binary_for_passk=true
bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn5_hf.sh JinyeopSong/KG-R1_test cwq \
  reward_model.reward_kwargs.binary_exact_match_mode=tog_substring


bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn5_hf.sh SCL2025/KG-R1-CQW cwq \
  reward_model.reward_kwargs.use_legacy_entity_f1=false \
  reward_model.reward_kwargs.use_exact_match_binary_for_passk=true \
  reward_model.reward_kwargs.exact_match_mismatch_jsonl="eval_results/eval_kg-r1/live_mismatch_cases.jsonl"

bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn5_hf.sh SCL2025/KG-R1-CQW cwq \
  reward_model.reward_kwargs.binary_exact_match_mode=tog_substring

bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn5_hf.sh SCL2025/KG-R1-CWQ-7B cwq \
  reward_model.reward_kwargs.binary_exact_match_mode=tog_substring

bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn5_hf.sh SCL2025/KG-R1-CQW cwq \
  reward_model.reward_kwargs.use_legacy_entity_f1=true \
  reward_model.reward_kwargs.use_exact_match_binary_for_passk=false

bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn5_hf.sh SCL2025/KG-R1-CWQ-hit1 cwq \
  reward_model.reward_kwargs.binary_exact_match_mode=tog_substring

bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn5_hf.sh SCL2025/KG-R1-WebQSP-hit1 webqsp \
  reward_model.reward_kwargs.binary_exact_match_mode=tog_substring

# Ablation
bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn5_hf.sh SCL2025/KG-R1-CWQ-no-turn-reward cwq \

bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn5_hf.sh SCL2025/KG-R1-CWQ-hit1-no-turn-advantage cwq \

bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn5_hf.sh SCL2025/KG-R1-CWQ-no-retrieval-reward cwq \

bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn5_hf.sh SCL2025/KG-R1-CWQ-hierarchical-retrieval cwq \

bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn5_hf.sh SCL2025/KG-R1-CWQ-hit1-PPO cwq \

bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn5_hf.sh SCL2025/KG-R1-CWQ-7B cwq 

bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn5_hf.sh SCL2025/KG-R1-CWQ-no-turn-reward webqsp 

CUDA_VISIBLE_DEVICES=0 bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn5_hf.sh SCL2025/KG-R1-CWQ-hit1-no-turn-advantage webqsp \
  --experiment_postfix=WebQSP-no-turn-advantage_2

CUDA_VISIBLE_DEVICES=1 bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn5_hf.sh SCL2025/KG-R1-CWQ-no-retrieval-reward webqsp \
  --experiment_postfix=WebQSP-no-retrieval-reward_2

CUDA_VISIBLE_DEVICES=2 bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn5_hf.sh SCL2025/KG-R1-CWQ-hierarchical-retrieval webqsp \
  --experiment_postfix=WebQSP-hierarchical-retrieval_2

CUDA_VISIBLE_DEVICES=3 bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn5_hf.sh SCL2025/KG-R1-CWQ-hit1-PPO webqsp \
  --experiment_postfix=WebQSP-PPO_2

CUDA_VISIBLE_DEVICES=0,1 bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn5_hf.sh SCL2025/KG-R1-CWQ-7B webqsp \
  --experiment_postfix=WebQSP-7B_2

##################

bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn5_hf.sh SCL2025/KG-R1-CQW webqsp \
  reward_model.reward_kwargs.use_legacy_entity_f1=false \
  reward_model.reward_kwargs.use_exact_match_binary_for_passk=true

bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn5_hf.sh SCL2025/KG-R1-CQW webqsp \
  reward_model.reward_kwargs.binary_exact_match_mode=tog_substring


##################

python -m baselines.vanilla \
  --dataset cwq \
  --num_workers 512 --experiment_name qwen2_7B-4

python -m baselines.cot \
  --dataset cwq \
  --num_workers 512 --experiment_name qwen2_7B-4

python -m baselines.sc \
  --dataset cwq \
  --num_workers 512 --experiment_name qwen2_7B-4

##################

python -m baselines.vanilla \
  --dataset webqsp \
  --num_workers 512 --experiment_name qwen2_7B-4

python -m baselines.cot \
  --dataset webqsp \
  --num_workers 512 --experiment_name qwen2_7B-4

python -m baselines.sc \
  --dataset webqsp \
  --num_workers 512 --experiment_name qwen2_7B-4


##################

python -m baselines.vanilla \
  --dataset cwq \
  --num_workers 512 --experiment_name qwen2_72B-4

python -m baselines.cot \
  --dataset cwq \
  --num_workers 512 --experiment_name qwen2_72B-4

python -m baselines.sc \
  --dataset cwq \
  --num_workers 512 --experiment_name qwen2_72B-4

##################

python -m baselines.vanilla \
  --dataset webqsp \
  --num_workers 512 --experiment_name qwen2_72B-4

python -m baselines.cot \
  --dataset webqsp \
  --num_workers 512 --experiment_name qwen2_72B-4

python -m baselines.sc \
  --dataset webqsp \
  --num_workers 512 --experiment_name qwen2_72B-4

##################

python -m baselines.vanilla \
  --dataset cwq \
  --num_workers 512 --experiment_name qwen3_235B-3

python -m baselines.cot \
  --dataset cwq \
  --num_workers 512 --experiment_name qwen3_235B-3

python -m baselines.sc \
  --dataset cwq \
  --num_workers 512 --experiment_name qwen3_235B-3

##################

python -m baselines.vanilla \
  --dataset webqsp \
  --num_workers 512 --experiment_name qwen3_235B-3

python -m baselines.cot \
  --dataset webqsp \
  --num_workers 512 --experiment_name qwen3_235B-3

python -m baselines.sc \
  --dataset webqsp \
  --num_workers 512 --experiment_name qwen3_235B-3

##################

python -m baselines.vanilla \
  --dataset qald \
  --num_workers 512 --experiment_name qwen3_235B-0

python -m baselines.cot \
  --dataset qald \
  --num_workers 512 --experiment_name qwen3_235B-0

python -m baselines.sc \
  --dataset qald \
  --num_workers 512 --experiment_name qwen3_235B-0

python -m baselines.vanilla \
  --dataset qald \
  --num_workers 512 --experiment_name qwen3_235B-1

python -m baselines.cot \
  --dataset qald \
  --num_workers 512 --experiment_name qwen3_235B-1

python -m baselines.sc \
  --dataset qald \
  --num_workers 512 --experiment_name qwen3_235B-1

python -m baselines.vanilla \
  --dataset qald \
  --num_workers 512 --experiment_name qwen3_235B-2

python -m baselines.cot \
  --dataset qald \
  --num_workers 512 --experiment_name qwen3_235B-2

python -m baselines.sc \
  --dataset qald \
  --num_workers 512 --experiment_name qwen3_235B-2

##################

python -m baselines.vanilla \
  --dataset qald \
  --num_workers 512 --experiment_name qwen2_7B-2

python -m baselines.cot \
  --dataset qald \
  --num_workers 512 --experiment_name qwen2_7B-2

python -m baselines.sc \
  --dataset qald \
  --num_workers 512 --experiment_name qwen2_7B-2