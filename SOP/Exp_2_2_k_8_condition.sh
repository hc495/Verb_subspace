export PATH=$PATH:~/anaconda3/bin
source activate ver
cd Verb_subspace

model_name="meta-llama/Meta-Llama-3-8B"
save_path="logs/Meta-Llama-3-8B_openend_k"

for dataset_index in 0 1 2 3 5 6
do
    python main_experiments.py \
        --model_name $model_name \
        --ICL_dataset_index $dataset_index \
        --huggingface_token "your_hf_token" \
        --injected_layer_num -1 \
        --injected_rank 0 \
        --bias "encoder" \
        --save_path "${save_path}_8_hidden_ood/ICL_${dataset_index}" \
        --num_epochs 0 \
        --train_part "none" \
        --open_end_test \
        --icl_demo_numbers 8 \
        --no_pre_test \
        --ood_test \
        --output_hidden_states
done

for dataset_index in 0 1 2 3 5 6
do
    python main_experiments.py \
        --model_name $model_name \
        --ICL_dataset_index $dataset_index \
        --huggingface_token "your_hf_token" \
        --injected_layer_num -1 \
        --injected_rank 0 \
        --bias "encoder" \
        --save_path "${save_path}_8_hidden_random_y/ICL_${dataset_index}" \
        --num_epochs 0 \
        --train_part "none" \
        --open_end_test \
        --icl_demo_numbers 8 \
        --no_pre_test \
        --random_label_word_test \
        --output_hidden_states
done