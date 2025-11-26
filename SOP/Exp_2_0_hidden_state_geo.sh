# Data processing: eccen_and_cov.ipynb
# Data processing should be run after all the Exp_2* scripts are finished.

export PATH=$PATH:~/anaconda3/bin
source activate ver
cd Verb_subspace

model_name="meta-llama/Meta-Llama-3-8B"
save_path="logs/Meta-Llama-3-8B_openend_k"
k_list=(0 1 2 4 8 16 32)

for k in "${k_list[@]}"
do
    for dataset_index in 0 1 2 3 5 6
    do
        python main_experiments.py \
            --model_name $model_name \
            --ICL_dataset_index $dataset_index \
            --huggingface_token "your_hf_token" \
            --injected_layer_num -1 \
            --injected_rank 0 \
            --bias "encoder" \
            --save_path "${save_path}_${k}_hidden/ICL_${dataset_index}" \
            --num_epochs 0 \
            --train_part "none" \
            --open_end_test \
            --icl_demo_numbers $k \
            --no_pre_test \
            --output_hidden_states 
    done
done

k_list=(64 128 256)

for k in "${k_list[@]}"
do
    for dataset_index in 0 1 2 3 5 6
    do
        python main_experiments.py \
            --model_name $model_name \
            --ICL_dataset_index $dataset_index \
            --huggingface_token "your_hf_token" \
            --injected_layer_num -1 \
            --injected_rank 0 \
            --bias "encoder" \
            --save_path "${save_path}_${k}_hidden/ICL_${dataset_index}" \
            --num_epochs 0 \
            --train_part "none" \
            --open_end_test \
            --icl_demo_numbers $k \
            --no_pre_test \
            --quantized \
            --output_hidden_states 
    done
done