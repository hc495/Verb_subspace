export PATH=$PATH:~/anaconda3/bin
source activate ver
cd Verb_subspace

model_name="meta-llama/Meta-Llama-3-8B"
save_path="logs/Meta-Llama-3-8B/"
layer_list=(0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 31)
rank_list=(0 1 2 4 8 16 32 2048 4096 8192)

for i in "${layer_list[@]}"
do
    for rank in "${rank_list[@]}"
    do
        python main_experiments.py \
            --model_name $model_name \
            --ICL_dataset_index -1 \
            --ICL_dataset_name "capital" \
            --huggingface_token "your_hf_token" \
            --injected_layer_num $i \
            --injected_rank $rank \
            --bias "encoder" \
            --save_path "${save_path}ICL_capital_${i}_${rank}" \
            --open_end_test
    done
done