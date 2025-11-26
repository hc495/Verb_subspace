import torch
import argparse
import os
from datetime import datetime
from util import my_model_kernel, functional
import pickle
from util import load_model_and_data as lmd
import StaICC

parser = argparse.ArgumentParser(description="LLM inference measurement")
parser.add_argument("--model_name", type=str, required=True, help="Path to the pretrained model")
parser.add_argument("--ICL_dataset_index", type=int, help="ICL dataset index")
parser.add_argument("--huggingface_token", type=str, help="Huggingface token for model access")
parser.add_argument("--quantized", action="store_true", help="Use quantized model")
parser.add_argument("--encoder_model_name", type=str, default="BAAI/bge-m3", help="Name of the encoder model")
parser.add_argument("--save_path", type=str, default="logs", help="Path to save the logs")

args = parser.parse_args()

# Load the pretrained model
ICL_model, ICL_tknz = lmd.load_ICL_model(args.model_name, huggingface_token = args.huggingface_token, quantized = args.quantized)
encoder_model, encoder_tknz = lmd.load_encode_model(args.encoder_model_name, huggingface_token = args.huggingface_token)
benchmark = StaICC.Normal(0)
experimentor = benchmark[args.ICL_dataset_index]
experimentor.prompt_former.replace_space_to_label()

prompts, queries = lmd.load_prompts_and_queries_from_StaICC_experimentor(experimentor)
ICL_hidden_states = my_model_kernel.ICL_inference_to_last_token_hidden_states(ICL_model, ICL_tknz, prompts)
encoder_feature = my_model_kernel.encoder_inference_to_feature(encoder_model, encoder_tknz, queries)

# Calculate the kernel alignment
res_kernel_alignment = functional.kernel_alignment_on_datasets(ICL_hidden_states, encoder_feature)

# Calculate the Scree Plot
scree_plot_res = []
pca_model = []
for layer_index in range(len(ICL_hidden_states)):
    varience_loaded, pca_res = functional.scree_plot_from_pca(ICL_hidden_states[layer_index])
    scree_plot_res.append(varience_loaded)
    pca_model.append(pca_res)

# Save the results
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
save_file = os.path.join(args.save_path, f"ICL_dataset_{args.model_name.replace('/', '_')}_index_{args.ICL_dataset_index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
with open(save_file, 'wb') as f:
    pickle.dump({
        "kernel_alignment": res_kernel_alignment,
        "scree_plot": scree_plot_res,
        "ICL_hidden_states": ICL_hidden_states,
        "encoder_feature": encoder_feature,
        "pca_model": pca_model,
    }, f)
print(f"Results saved to {save_file}")
# Also save the results as a txt file
txt_save_file = os.path.splitext(save_file)[0] + ".txt"
with open(txt_save_file, 'w') as f:
    f.write("Kernel Alignment:\n")
    f.write(str(res_kernel_alignment) + "\n\n")
    f.write("Scree Plot Results:\n")
    for idx, scree in enumerate(scree_plot_res):
        f.write(f"Layer {idx}: {scree}\n")
print(f"Results also saved to {txt_save_file}")