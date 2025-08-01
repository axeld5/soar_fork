# tuto for launching experiment

## install env

First install the both conda (check readme for more details):
- Inference env `sglang47` (you can upgrade to lateset SGLang version) 
- Training env `unsloth_env`


Now fist let's set path variables

## init variables

```bash
export work_path="/home/flowers/work"
export soar_path="$work_path/SOAR"
export model_id_base="Qwen2.5-Coder-3B-Instruct" # model that you want to use here
export path_base_model="$work_path/hf/$model_id_base"
export path_embed_model="$work_path/hf/CodeRankEmbed"
export split="train" # "train" -> public test set  or "val" -> public test set 

# n_sample_task -> number of task sample per task to finetuned the model (should be 50 but for demo set to 5)
export n_sample_task=50
export model_id="${model_id_base}_n${n_sample_task}"
```

## LOOP start here 

Choose generation number:

```bash
# generation number strat at 0 then 1,2,...

export gen="0"

if [ "$gen" -eq 0 ]; then
    # use base model for generation 0
    export path_model="$work_path/hf/$model_id_base" 
    
else
    # use finetuned model for other generation
    export path_model="$work_path/hf/trained/$model_id_base/gen-$gen/$model_id"
fi
```

### sampling intial solution

To sample the 3k initial solution (here -k 2 for demo, change that to 3000 for full run) run this:

```bash
conda deactivate
conda activate sglang47

# Sample phase (default fewshot example from train_solution.pkl but should be from train archive of previous generation)

python $soar_path/soar/inference/sample_phase.py \
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
```

Note: you can increase n_gpu to the number of gpu you have to make things faster and to use LLMs that don't fit on one GPUs (used 1 GPUs for 7b, 2 GPUs for 14b and 4 GPUs for LLMs with more than 14b params)

Then do post-processing on those data (rm code with error, ...)

```bash
# Process results sample phase
python $soar_path/soar/post_process/merge_filter.py \
--base_path $soar_path/ \
--path_folder $soar_path/save_results/$model_id_base/gen-$gen/ \
--max_data 3000 \
--split $split 
```


### Refine initial solution

Now we need to refine those initial solutions. To save time you can run the 5 seed on separate nodes (each seed is 600 refined codes)

So execute this:
```bash
for seed in {0..4}; do
    python $soar_path/soar/repair/rex_inference.py \
    --base_path $soar_path/ \
    --path_model $path_model/ \
    --path_archive $soar_path/save_results/$model_id_base/gen-$gen/solution/train_sol.pkl \
    --model_len 30000 \
    --total_budget 600 \
    --seed $seed \
    --split $split \
    --max_tokens 2048 \
    --n_completion 1 \
    --gpu_mem 0.87 \
    --n_gpu 1
done
```

Or to get the results faster if you have multiple nodes, this:

```bash
# run this for five different seed (0,1,2,3,4) in pa
export seed=0

python $soar_path/soar/repair/rex_inference.py \
--base_path $soar_path/ \
--path_model $path_model/ \
--path_archive $soar_path/save_results/$model_id_base/gen-$gen/solution/train_sol.pkl \
--model_len 30000 \
--total_budget 600 \
--seed $seed \
--split $split \
--max_tokens 2048 \
--n_completion 1 \
--gpu_mem 0.87 \
--n_gpu 1
```


Then do post-processing on those data (rm code with error, ...)

```bash
python $soar_path/soar/post_process/merge_filter.py \
--base_path $soar_path/ \
--path_folder $soar_path/save_results/$model_id_base/gen-$gen/ \
--max_data 3000 \
--split $split \
--use_repair_data
```

Optional merge results and deduplication based on code in one results (for merging responses from all models in the paper)
```bash
python $soar_path/soar/post_process/dedup.py \
--base_path $soar_path/ \
--path_embed_model $path_embed_model/ \
--path_save $soar_path/save_results/$model_id_base/gen-$gen/solution/train_sol_repair.pkl \
--path_0 $soar_path/save_results/$model_id_base/gen-$gen/solution/train_sol_repair.pkl 
```

### Sample data for training model (gen + 1)-th generation

For sampling data to improving model on sampling intial solution:

```bash
# sample training data for sample training (use train_sol_repair.pkl in path_folder_solution as data)
python $soar_path/soar/post_process/process_sample_for_training.py \
--path_model $path_model/ \
--path_embed_model $path_embed_model/ \
--split $split \
--N_sample_task $n_sample_task \
--path_folder_solution $soar_path/save_results/$model_id_base/gen-$gen/solution/ \
--sampling_her greedy_div 
```


For sampling data to improving model on refinement

```bash
# sample training data for sample refinement (use every data starting with $split in path_folder_refinement as data)
python $soar_path/soar/post_process/process_repair_for_training.py \
--base_path $soar_path/ \
--path_embed_model $path_embed_model/ \
--split $split \
--N_sample_task $n_sample_task \
--path_folder_refinement $soar_path/save_results/$model_id_base/gen-$gen/refinement/
```


### Training the model


```bash
# Train model (add --load_in_4bit for Q-LoRA)

conda deactivate
conda activate unsloth_env
python $soar_path/soar/training/train_unsloth.py \
--base_path $soar_path/ \
--model_len 30000 \
--path_base_model $path_base_model \
--path_archive $soar_path/save_results/$model_id_base/gen-$gen/solution/data4train-train-n$n_sample_task-greedy_div.pkl \
--path_archive_repair $soar_path/save_results/$model_id_base/gen-$gen/refinement/data4train-train-n$n_sample_task.pkl \
--path_save_model $work_path/hf/trained/$model_id_base/gen-$((gen + 1))/$model_id
```

## LOOP

