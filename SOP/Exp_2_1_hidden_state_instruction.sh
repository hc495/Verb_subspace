export PATH=$PATH:~/anaconda3/bin
source activate ver
cd Verb_subspace

model_name="meta-llama/Meta-Llama-3-8B"
save_path_clean="logs/Meta-Llama-3-8B_openend_k_0_instruction_hidden/clean/"
save_path_label="logs/Meta-Llama-3-8B_openend_k_0_instruction_hidden/label/"

for dataset_index in 0 
do
    python main_experiments.py \
        --model_name $model_name \
        --ICL_dataset_index $dataset_index \
        --huggingface_token "your_hf_token" \
        --injected_layer_num -1 \
        --injected_rank 0 \
        --bias "encoder" \
        --save_path "${save_path_clean}ICL_${dataset_index}" \
        --num_epochs 0 \
        --train_part "none" \
        --open_end_test \
        --icl_demo_numbers 0 \
        --no_pre_test \
        --output_hidden_states \
        --instruction "You are a helpful assistant. Please predict the sentiment of the following sentence: " 
done

for dataset_index in 0 
do
    python main_experiments.py \
        --model_name $model_name \
        --ICL_dataset_index $dataset_index \
        --huggingface_token "your_hf_token" \
        --injected_layer_num -1 \
        --injected_rank 0 \
        --bias "encoder" \
        --save_path "${save_path_label}ICL_${dataset_index}" \
        --num_epochs 0 \
        --train_part "none" \
        --open_end_test \
        --icl_demo_numbers 0 \
        --no_pre_test \
        --output_hidden_states \
        --instruction "You are a helpful assistant. Please predict the sentiment of the following sentence in positive and negative: " 
done

for dataset_index in 1
do
    python main_experiments.py \
        --model_name $model_name \
        --ICL_dataset_index $dataset_index \
        --huggingface_token "your_hf_token" \
        --injected_layer_num -1 \
        --injected_rank 0 \
        --bias "encoder" \
        --save_path "${save_path_clean}ICL_${dataset_index}" \
        --num_epochs 0 \
        --train_part "none" \
        --open_end_test \
        --icl_demo_numbers 0 \
        --no_pre_test \
        --output_hidden_states \
        --instruction "You are a helpful assistant. Please predict the sentiment of the following sentence: " 
done

for dataset_index in 1
do
    python main_experiments.py \
        --model_name $model_name \
        --ICL_dataset_index $dataset_index \
        --huggingface_token "your_hf_token" \
        --injected_layer_num -1 \
        --injected_rank 0 \
        --bias "encoder" \
        --save_path "${save_path_label}ICL_${dataset_index}" \
        --num_epochs 0 \
        --train_part "none" \
        --open_end_test \
        --icl_demo_numbers 0 \
        --no_pre_test \
        --output_hidden_states \
        --instruction "You are a helpful assistant. Please predict the sentiment of the following sentence in positive and negative: " 
done

for dataset_index in 2
do
    python main_experiments.py \
        --model_name $model_name \
        --ICL_dataset_index $dataset_index \
        --huggingface_token "your_hf_token" \
        --injected_layer_num -1 \
        --injected_rank 0 \
        --bias "encoder" \
        --save_path "${save_path_clean}ICL_${dataset_index}" \
        --num_epochs 0 \
        --train_part "none" \
        --open_end_test \
        --icl_demo_numbers 0 \
        --no_pre_test \
        --output_hidden_states \
        --instruction "You are a helpful assistant. Please predict the sentiment of the following sentence: " 
done

for dataset_index in 2
do
    python main_experiments.py \
        --model_name $model_name \
        --ICL_dataset_index $dataset_index \
        --huggingface_token "your_hf_token" \
        --injected_layer_num -1 \
        --injected_rank 0 \
        --bias "encoder" \
        --save_path "${save_path_label}ICL_${dataset_index}" \
        --num_epochs 0 \
        --train_part "none" \
        --open_end_test \
        --icl_demo_numbers 0 \
        --no_pre_test \
        --output_hidden_states \
        --instruction "You are a helpful assistant. Please predict the sentiment of the following sentence in positive, neutral, and negative: " 
done

for dataset_index in 3
do
    python main_experiments.py \
        --model_name $model_name \
        --ICL_dataset_index $dataset_index \
        --huggingface_token "your_hf_token" \
        --injected_layer_num -1 \
        --injected_rank 0 \
        --bias "encoder" \
        --save_path "${save_path_clean}ICL_${dataset_index}" \
        --num_epochs 0 \
        --train_part "none" \
        --open_end_test \
        --icl_demo_numbers 0 \
        --no_pre_test \
        --output_hidden_states \
        --instruction "You are a helpful assistant. Please predict the sentiment of the following sentence: " 
done


for dataset_index in 3
do
    python main_experiments.py \
        --model_name $model_name \
        --ICL_dataset_index $dataset_index \
        --huggingface_token "your_hf_token" \
        --injected_layer_num -1 \
        --injected_rank 0 \
        --bias "encoder" \
        --save_path "${save_path_label}ICL_${dataset_index}" \
        --num_epochs 0 \
        --train_part "none" \
        --open_end_test \
        --icl_demo_numbers 0 \
        --no_pre_test \
        --output_hidden_states \
        --instruction "You are a helpful assistant. Please predict the sentiment of the following sentence in poor, bad, neutral, good, and great: " 
done

for dataset_index in 5 
do
    python main_experiments.py \
        --model_name $model_name \
        --ICL_dataset_index $dataset_index \
        --huggingface_token "your_hf_token" \
        --injected_layer_num -1 \
        --injected_rank 0 \
        --bias "encoder" \
        --save_path "${save_path_clean}ICL_${dataset_index}" \
        --num_epochs 0 \
        --train_part "none" \
        --open_end_test \
        --icl_demo_numbers 0 \
        --no_pre_test \
        --output_hidden_states \
        --instruction "You are a helpful assistant. Please predict the category of this news: " 
done


for dataset_index in 5 
do
    python main_experiments.py \
        --model_name $model_name \
        --ICL_dataset_index $dataset_index \
        --huggingface_token "your_hf_token" \
        --injected_layer_num -1 \
        --injected_rank 0 \
        --bias "encoder" \
        --save_path "${save_path_label}ICL_${dataset_index}" \
        --num_epochs 0 \
        --train_part "none" \
        --open_end_test \
        --icl_demo_numbers 0 \
        --no_pre_test \
        --output_hidden_states \
        --instruction "You are a helpful assistant. Please predict the category of this news in world, sports, business, science: " 
done


for dataset_index in 6
do
    python main_experiments.py \
        --model_name $model_name \
        --ICL_dataset_index $dataset_index \
        --huggingface_token "your_hf_token" \
        --injected_layer_num -1 \
        --injected_rank 0 \
        --bias "encoder" \
        --save_path "${save_path_clean}ICL_${dataset_index}" \
        --num_epochs 0 \
        --train_part "none" \
        --open_end_test \
        --icl_demo_numbers 0 \
        --no_pre_test \
        --output_hidden_states \
        --instruction "You are a helpful assistant. Please predict the subjectivity of this sentence: " 
done

for dataset_index in 6
do
    python main_experiments.py \
        --model_name $model_name \
        --ICL_dataset_index $dataset_index \
        --huggingface_token "your_hf_token" \
        --injected_layer_num -1 \
        --injected_rank 0 \
        --bias "encoder" \
        --save_path "${save_path_label}ICL_${dataset_index}" \
        --num_epochs 0 \
        --train_part "none" \
        --open_end_test \
        --icl_demo_numbers 0 \
        --no_pre_test \
        --output_hidden_states \
        --instruction "You are a helpful assistant. Please predict the subjectivity of this sentence in objective and subjective: " 
done