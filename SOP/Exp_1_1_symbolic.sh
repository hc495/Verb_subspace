export PATH=$PATH:~/anaconda3/bin
source activate ver
cd Verb_subspace

model_name="meta-llama/Meta-Llama-3-8B"
save_path="logs/Meta-Llama-3-8B_openend_symb/"
trained_autoencoder_path="logs/Llama-3.2-1B_openend/ICL_0_8_8/Llama-3.2-1B_llama-3_8_8_none_encoder_20250527_233206_173.pkl"

for i in 8
do
    for rank in 8
    do
        for dataset_index in 0
        do
            python main_experiments.py \
                --model_name $model_name \
                --ICL_dataset_index $dataset_index \
                --huggingface_token "your_hf_token" \
                --injected_layer_num $i \
                --injected_rank $rank \
                --bias "encoder" \
                --save_path $save_path"ICL_${dataset_index}_${i}_${rank}_decoder" \
                --symbolic_label \
                --num_epochs 10 \
                --trained_autoencoder $trained_autoencoder_path \
                --train_part "decoder" \
                --open_end_test
        done
    done
done

for i in 8
do
    for rank in 8
    do
        for dataset_index in 0
        do
            python main_experiments.py \
                --model_name $model_name \
                --ICL_dataset_index $dataset_index \
                --huggingface_token "your_hf_token" \
                --injected_layer_num $i \
                --injected_rank $rank \
                --bias "encoder" \
                --save_path $save_path"ICL_${dataset_index}_${i}_${rank}_encoder" \
                --symbolic_label \
                --num_epochs 10 \
                --trained_autoencoder $trained_autoencoder_path \
                --train_part "encoder" \
                --open_end_test
        done
    done
done

for i in 8
do
    for rank in 8
    do
        for dataset_index in 0
        do
            python main_experiments.py \
                --model_name $model_name \
                --ICL_dataset_index $dataset_index \
                --huggingface_token "your_hf_token" \
                --injected_layer_num $i \
                --injected_rank $rank \
                --bias "encoder" \
                --save_path $save_path"ICL_${dataset_index}_${i}_${rank}_both" \
                --symbolic_label \
                --num_epochs 10 \
                --trained_autoencoder $trained_autoencoder_path \
                --train_part "both" \
                --open_end_test
        done
    done
done