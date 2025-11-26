export PATH=$PATH:~/anaconda3/bin
source activate ver
cd Verb_subspace

model_name="meta-llama/Meta-Llama-3-8B"

python encoding_magnitude_and_inner_dimensions.py \
    --model_name $model_name \
    --ICL_dataset_index 0 \
    --huggingface_token "your_hf_token"

python encoding_magnitude_and_inner_dimensions.py \
    --model_name $model_name \
    --ICL_dataset_index 1 \
    --huggingface_token "your_hf_token"

python encoding_magnitude_and_inner_dimensions.py \
    --model_name $model_name \
    --ICL_dataset_index 2 \
    --huggingface_token "your_hf_token"

python encoding_magnitude_and_inner_dimensions.py \
    --model_name $model_name \
    --ICL_dataset_index 3 \
    --huggingface_token "your_hf_token"

python encoding_magnitude_and_inner_dimensions.py \
    --model_name $model_name \
    --ICL_dataset_index 5 \
    --huggingface_token "your_hf_token"

python encoding_magnitude_and_inner_dimensions.py \
    --model_name $model_name \
    --ICL_dataset_index 6 \
    --huggingface_token "your_hf_token"