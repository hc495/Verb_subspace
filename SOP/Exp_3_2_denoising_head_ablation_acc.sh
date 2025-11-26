model_name="meta-llama/Llama-3.2-1B"
save_path="logs/Llama-3.2-1B_openend_k_8_denoising_ablated"
head_stat_path="logs/visualization/Llama-3.2-1B_head_ablation_final_res"

for dataset_index in 0 1 2 3 5 6
do
    induction_heads_file_argu="${head_stat_path}/ICL_$dataset_index/denoising_heads.txt"
    induction_heads_str_argu=$(cat "$induction_heads_file_argu")
    echo "ablation heads: $induction_heads_str_argu"
    python main_experiments.py \
        --model_name "$model_name" \
        --ICL_dataset_index $dataset_index \
        --huggingface_token "your_hf_token" \
        --injected_layer_num -1 \
        --injected_rank 0 \
        --bias "encoder" \
        --save_path "${save_path}/ICL_${dataset_index}" \
        --num_epochs 0 \
        --train_part "none" \
        --icl_demo_numbers 8 \
        --open_end_test \
        --no_pre_test \
        --random_ablate_heads \
        --ablated_heads "$induction_heads_str_argu"

    python main_experiments.py \
        --model_name "$model_name" \
        --ICL_dataset_index $dataset_index \
        --huggingface_token "your_hf_token" \
        --injected_layer_num -1 \
        --injected_rank 0 \
        --bias "encoder" \
        --save_path "${save_path}/ood/ICL_${dataset_index}" \
        --num_epochs 0 \
        --train_part "none" \
        --icl_demo_numbers 8 \
        --open_end_test \
        --ood_test \
        --no_pre_test \
        --random_ablate_heads \
        --ablated_heads "$induction_heads_str_argu"

    python main_experiments.py \
        --model_name "$model_name" \
        --ICL_dataset_index $dataset_index \
        --huggingface_token "your_hf_token" \
        --injected_layer_num -1 \
        --injected_rank 0 \
        --bias "encoder" \
        --save_path "${save_path}/id/ICL_${dataset_index}" \
        --num_epochs 0 \
        --train_part "none" \
        --icl_demo_numbers 8 \
        --open_end_test \
        --id_test \
        --no_pre_test \
        --random_ablate_heads \
        --ablated_heads "$induction_heads_str_argu"

    python main_experiments.py \
        --model_name "$model_name" \
        --ICL_dataset_index $dataset_index \
        --huggingface_token "your_hf_token" \
        --injected_layer_num -1 \
        --injected_rank 0 \
        --bias "encoder" \
        --save_path "${save_path}/ICL_${dataset_index}" \
        --num_epochs 0 \
        --train_part "none" \
        --icl_demo_numbers 8 \
        --open_end_test \
        --no_pre_test \
        --ablated_heads "$induction_heads_str_argu"

    python main_experiments.py \
        --model_name "$model_name" \
        --ICL_dataset_index $dataset_index \
        --huggingface_token "your_hf_token" \
        --injected_layer_num -1 \
        --injected_rank 0 \
        --bias "encoder" \
        --save_path "${save_path}/ood/ICL_${dataset_index}" \
        --num_epochs 0 \
        --train_part "none" \
        --icl_demo_numbers 8 \
        --open_end_test \
        --ood_test \
        --no_pre_test \
        --ablated_heads "$induction_heads_str_argu"

    python main_experiments.py \
        --model_name "$model_name" \
        --ICL_dataset_index $dataset_index \
        --huggingface_token "your_hf_token" \
        --injected_layer_num -1 \
        --injected_rank 0 \
        --bias "encoder" \
        --save_path "${save_path}/id/ICL_${dataset_index}" \
        --num_epochs 0 \
        --train_part "none" \
        --icl_demo_numbers 8 \
        --open_end_test \
        --id_test \
        --no_pre_test \
        --ablated_heads "$induction_heads_str_argu"
done