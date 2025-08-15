# download models (if needed)
huggingface-cli download Qwen/Qwen2.5-Coder-3B-Instruct --local-dir Qwen2.5-Coder-3B-Instruct --local-dir-use-symlinks True --cache-dir ./hf_cache
huggingface-cli download nomic-ai/CodeRankEmbed --local-dir CodeRankEmbed --local-dir-use-symlinks True --cache-dir ./hf_cache


# init variables (adjust paths for your system)
export work_path="${WORK_PATH:-$(pwd)}"  # Use current directory if WORK_PATH not set
export soar_path="$work_path"
export model_id_base="Qwen2.5-Coder-3B-Instruct" # "Qwen3-1.7B" "Qwen2.5-Coder-3B-Instruct" #
export path_base_model="$work_path/hf/$model_id_base"
export path_embed_model="$work_path/hf/CodeRankEmbed"
export split="train" # "train" "val"
export n_sample_task=50
export model_id="${model_id_base}_n${n_sample_task}"
# generation (0,1,2,3,...) 
export gen="0"


# if gen is 0, we use the base model
if [ "$gen" -eq 0 ]; then
    export path_model="$work_path/hf/$model_id_base"
    
else
    export path_model="$work_path/hf/trained/$model_id_base/gen-$gen/$model_id"
fi


# Sample phase (default fewshot example from train_solution.pkl but should be from train archive of previous generation)
# Using UV inference environment
uv run --extra inference python $soar_path/soar/inference/sample_phase.py \
--base_path $soar_path/ \
--path_model $path_model/ \
--path_save_res $soar_path/save_results/$model_id_base/gen-$gen/solution \
-k 3000 \
--model_len 30000 \
--gpu_mem 0.89 \
--n_gpu 1 \
--split $split \
--bs_inference 1 \
--seed 0 \
--max_tokens 4096 \
--path_fewshot $soar_path/soar/inference/train_solutions.pkl

# Process results sample phase
uv run python $soar_path/soar/post_process/merge_filter.py \
--base_path $soar_path/ \
--path_folder $soar_path/save_results/$model_id_base/gen-$gen/ \
--max_data 3000 \
--split $split 

# Refinement phase
for seed in {0..4}; do
    uv run --extra inference python $soar_path/soar/repair/rex_inference.py \
    --base_path $soar_path/ \
    --path_model $path_model/ \
    --path_archive $soar_path/save_results/$model_id_base/gen-$gen/solution/train_sol.pkl \
    --model_len 30000 \
    --total_budget 600 \
    --seed $seed \
    --split $split \
    --max_tokens 2048 \
    --n_completion 1 \
    --gpu_mem 0.87
done

# Process results sample&refinement phase
uv run python $soar_path/soar/post_process/merge_filter.py \
--base_path $soar_path/ \
--path_folder $soar_path/save_results/$model_id_base/gen-$gen/ \
--max_data 3000 \
--split $split \
--use_repair_data

# optional merge results and deduplication in one results
uv run python $soar_path/soar/post_process/dedup.py \
--base_path $soar_path/ \
--path_embed_model $path_embed_model/ \
--path_save $soar_path/save_results/$model_id_base/gen-$gen/solution/train_sol_repair.pkl \
--path_0 $soar_path/save_results/$model_id_base/gen-$gen/solution/train_sol_repair.pkl 


# sample training data for sample training (use train_sol_repair.pkl in path_folder_solution as data)
uv run python $soar_path/soar/post_process/process_sample_for_training.py \
--path_model $path_model/ \
--path_embed_model $path_embed_model/ \
--split $split \
--N_sample_task $n_sample_task \
--path_folder_solution $soar_path/save_results/$model_id_base/gen-$gen/solution/ \
--sampling_her greedy_div 

# sample training data for sample refinement (use every data starting with $split in path_folder_refinement as data)
uv run python $soar_path/soar/post_process/process_repair_for_training.py \
--base_path $soar_path/ \
--path_embed_model $path_embed_model/ \
--split $split \
--N_sample_task $n_sample_task \
--path_folder_refinement $soar_path/save_results/$model_id_base/gen-$gen/refinement/


# Train model (add --load_in_4bit for Q-LoRA)
# Using UV training environment
uv run --extra training python $soar_path/soar/training/train_unsloth.py \
--base_path $soar_path/ \
--model_len 30000 \
--path_base_model $path_base_model \
--path_archive $soar_path/save_results/$model_id_base/gen-$gen/solution/data4train-train-n$n_sample_task-greedy_div.pkl \
--path_archive_repair $soar_path/save_results/$model_id_base/gen-$gen/refinement/data4train-train-n$n_sample_task.pkl \
--path_save_model $work_path/hf/trained/$model_id_base/gen-$((gen + 1))/$model_id
# Loop

