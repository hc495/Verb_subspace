# Should also run the Exp4_Induction_Head_subspace_visualization.ipynb for the final visualization in head_ablation_visualization.ipynb.
# Warning: The output of this script is HUGE, so it is recommended to run it on a machine with sufficient disk space.

export PATH=$PATH:~/anaconda3/bin
source activate ver
cd Verb_subspace

layer_start=8
max_layer=24
layer_jump=1
max_heads=31
head_jump=1
model_name="meta-llama/Meta-Llama-3-8B"
save_path="logs/Meta-Llama-3-8B_openend_k_8_hidden_ablated"

for layer in $(seq $layer_start $layer_jump $max_layer)
do
    for dataset_index in 0 1 2 3 5 6 
    do
        for ablated_heads in $(seq 0 $head_jump $max_heads)
        do
            python main_experiments.py \
                --model_name $model_name \
                --ICL_dataset_index $dataset_index \
                --huggingface_token "your_hf_token" \
                --injected_layer_num -1 \
                --injected_rank 0 \
                --bias "encoder" \
                --save_path "$save_path/ablated_layer_${layer}_${ablated_heads}/ICL_${dataset_index}" \
                --num_epochs 0 \
                --train_part "none" \
                --open_end_test \
                --icl_demo_numbers 8 \
                --no_pre_test \
                --output_hidden_states \
                --ablated_heads "{${layer}: [${ablated_heads}]}" 
        done
    done
done