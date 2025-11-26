# Visualization: logs/visualization/Exp1.ipynb

export PATH=$PATH:~/anaconda3/bin
source activate ver
cd Verb_subspace

model_name="meta-llama/Meta-Llama-3-8B"
save_path="logs/Meta-Llama-3-8B/"
layer_list=(0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 31)
rank_list=(0 1 2 4 8 16 32 2048)

for dataset_index in 0 1 2 3 5 6
do
    for i in "${layer_list[@]}"
    do
        for rank in "${rank_list[@]}"
        do
            python main_experiments.py \
                --model_name $model_name \
                --ICL_dataset_index $dataset_index \
                --huggingface_token "your_hf_token" \
                --injected_layer_num $i \
                --injected_rank $rank \
                --bias "encoder" \
                --save_path $save_path"ICL_${dataset_index}_${i}_${rank}" \
                --open_end_test
        done
    done
done