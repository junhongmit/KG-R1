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


bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn5_hf.sh SCL2025/KG-R1-CWQ-7B cwq \
  reward_model.reward_kwargs.use_legacy_entity_f1=false \
  reward_model.reward_kwargs.use_exact_match_binary_for_passk=true

bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn5_hf.sh SCL2025/KG-R1-CWQ-no-turn-reward cwq \
  reward_model.reward_kwargs.use_legacy_entity_f1=false \
  reward_model.reward_kwargs.use_exact_match_binary_for_passk=true

##################

bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn5_hf.sh SCL2025/KG-R1-CQW webqsp \
  reward_model.reward_kwargs.use_legacy_entity_f1=false \
  reward_model.reward_kwargs.use_exact_match_binary_for_passk=true

bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_turn5_hf.sh SCL2025/KG-R1-CQW webqsp \
  reward_model.reward_kwargs.binary_exact_match_mode=tog_substring

##################

python -m baselines.vanilla \
  --dataset cwq \
  --num_workers 512

python -m baselines.cot \
  --dataset cwq \
  --num_workers 512

python -m baselines.sc \
  --dataset cwq \
  --num_workers 512

##################

  python -m baselines.vanilla \
    --dataset webqsp \
    --num_workers 512

  python -m baselines.cot \
    --dataset webqsp \
    --num_workers 512

  python -m baselines.sc \
    --dataset webqsp \
    --num_workers 512