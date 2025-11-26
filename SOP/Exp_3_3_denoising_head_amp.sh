model_name="meta-llama/Llama-3.2-1B"
save_path="logs/Llama-3.2-1B_openend_k_8_denoising_amp"
head_stat_path="logs/visualization/Llama-3.2-1B_head_ablation_final_res"

for dataset_index in 0 1 2 3 5 6
do
    denoising_head_file="${head_stat_path}/ICL_$dataset_index/denoising_heads.txt"
    denoising_head_str=$(cat "$denoising_head_file")
    echo "heads: $denoising_head_str"
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
        --amplified_head "$denoising_head_str" \
        --amplify_factor 0.1