import torch
from StaICC.prefabricate_inference import model_kernel as model_kernel
from functools import partial
from injection_inference.llama3 import Llama3_injected
from injection_inference.qwen2 import Qwen2_injected
from injection_inference.autoencoder import autoencoder
from util import load_model_and_data as lmd
import StaICC
from util.trainer import Trainer
from util.my_model_kernel import standard_ICL_inference_with_injected
from util.my_model_kernel import test_model_on_staicc_experimentor, test_model
from util.functional import linear_regression
import wandb
from peft import prepare_model_for_kbit_training 
import ast

import argparse
import os
from datetime import datetime
import pickle

from util import my_dataset
import random
import glob

DATASET_NAME_TO_DATASET_CLASS = {
    "capital": my_dataset.capital,
    "profession": my_dataset.famous_people_prof,
    "translation": my_dataset.translation,
}

parser = argparse.ArgumentParser(description="Autoencoder injection inference")
parser.add_argument("--model_name", type=str, required=True, help="Path to the pretrained model")
parser.add_argument("--ICL_dataset_index", type=int, default=-1, help="ICL dataset index, defined in StaICC. If -1, use ICL_dataset_name instead.")
parser.add_argument("--ICL_dataset_name", type=str, default=None, help="ICL dataset name (only used when ICL_dataset_index is -1). Selection: 'capital', 'profession', 'translation'")
parser.add_argument("--huggingface_token", type=str, default=None, help="Huggingface token for model access. Empty to use os.environ['HF_TOKEN'] or no token.")
parser.add_argument("--residual", action="store_true", help="Use residual connection in the injected filter")
parser.add_argument("--quantized", action="store_true", help="Use quantized model")
parser.add_argument("--injected_rank", type=int, default=16, help="Rank of the injected filter")
parser.add_argument("--num_epochs", type=int, default=4, help="Number of training epochs")
parser.add_argument("--injected_activation", type=str, default="none", help="Activation function for the injected filter")
parser.add_argument("--injected_layer_num", type=int, default=0, help="Layer number to inject the filter into")
parser.add_argument("--bias", type=str, default="both", help="Bias for the injected filter (encoder, decoder, both, none)")
parser.add_argument("--save_path", type=str, default="logs", help="Path to save the logs")
parser.add_argument("--open_end_test", action="store_true", help="Whether to perform open-end test")
parser.add_argument("--symbolic_label", action="store_true", help="Whether to use symbolic label")
parser.add_argument("--trained_autoencoder", type=str, default=None, help="Path to a trained autoencoder to load")
parser.add_argument("--train_part", type=str, default="both", help="Train on encoder, decoder, none, or both (encoder, decoder, both)")
parser.add_argument("--icl_demo_numbers", type=int, default=0, help="Number of ICL demos to use for training and testing")
parser.add_argument("--ood_test", action="store_true", help="Whether to perform OOD test. Only works for StaICC datasets.")
parser.add_argument("--id_test", action="store_true", help="Whether to perform in-domain test. Only works for StaICC datasets.")
parser.add_argument("--random_label_word_test", action="store_true", help="Whether to perform random label (noisy label) test. Only works for StaICC datasets.")
parser.add_argument("--hook", action="store_true")
parser.add_argument("--pre_test_only", action="store_true", help="Whether to only perform pre-test without training")
parser.add_argument("--no_pre_test", action="store_true", help="Whether to skip pre-test and directly train the model")
parser.add_argument("--output_hidden_states", action="store_true", help="Whether to output hidden states from the model")
parser.add_argument("--output_full_hidden_states", action="store_true", help="Whether to output hidden states from the model")
parser.add_argument("--output_attentions", action="store_true", help="Whether to output attentions from the model")
parser.add_argument("--instruction", type=str, default=None, help="Instruction to use for the model (if applicable). Typically space needed in the end.")
parser.add_argument("--ablated_heads", type=ast.literal_eval, default={}, help="Heads to ablate in the model. Format: {'layer_num': [head1, head2, ...]}.")
parser.add_argument("--random_ablate_heads", action="store_true", help="Whether to randomly ablate heads in the model, with the same layer-wise amount as specified in --ablated_heads.")
parser.add_argument("--ablate_last_label", action="store_true", help="Whether to ablate the last label in the prompts. Only works for StaICC datasets.")
parser.add_argument("--estimate_filter", action="store_true", help="Whether to estimate the filter size based on the model's hidden states. Will cover the trained_autoencoder.")
parser.add_argument("--saved_zero_shot_hs", type=str, help="0 shot hidden states for the filter estimation")
parser.add_argument("--saved_few_shot_hs", type=str, help="few-shot hidden states for the filter estimation")
parser.add_argument("--amplify_factor", type=float, default=0.1, help="Amplification factor for the amplified_head.")
parser.add_argument("--amplified_head", type=ast.literal_eval, default={}, help="Head to amplify in the model. Format: {'layer_num': head_num}.")

args = parser.parse_args()

if args.huggingface_token is None:
    try:
        args.huggingface_token = os.environ['HF_TOKEN']
    except KeyError:
        print("Huggingface token must be provided either through --huggingface_token or HF_TOKEN environment variable. Use empty string in default.")
        args.huggingface_token = ""

############################# 
# Load the pretrained model
ICL_model, ICL_tknz = lmd.load_ICL_model(args.model_name, huggingface_token = args.huggingface_token, quantized = args.quantized)
if args.quantized:
    print("Preparing model for k-bit training...")
    ICL_model = prepare_model_for_kbit_training(ICL_model)
auto_encoder = autoencoder(
    input_dim = ICL_model.config.hidden_size,
    hidden_dim = args.injected_rank,
    output_dim = ICL_model.config.hidden_size,
    activation = "none",
    bias = args.bias,
    residual = args.residual,
    quantized = False # Always False for now, as we used prepare_model_for_kbit_training
)

if args.estimate_filter:
    ############################# 
    # Unused. 
    print("Estimating filter...")
    if args.trained_autoencoder:
        print("Warning: --trained_autoencoder will be ignored as --estimate_filter is set.")
    zero_shot_hs = []
    few_shot_hs = []
    def get_pkl_file(path):
        if os.path.isdir(path):
            pkl_files = glob.glob(os.path.join(path, "*.pkl"))
            if not pkl_files:
                raise FileNotFoundError(f"No .pkl files found in directory {path}")
            return pkl_files[0]
        return path

    zero_shot_hs_path = get_pkl_file(args.saved_zero_shot_hs)
    few_shot_hs_path = get_pkl_file(args.saved_few_shot_hs)
    with open(zero_shot_hs_path, "rb") as f:
        zero_shot_hs_file = pickle.load(f)
        for sample_index in range(len(zero_shot_hs_file['post_test_res']['hidden_states'])):
            zero_shot_hs.append(zero_shot_hs_file['post_test_res']['hidden_states'][sample_index][args.injected_layer_num][0][-1])
    with open(few_shot_hs_path, "rb") as f:
        few_shot_hs_file = pickle.load(f)
        for sample_index in range(len(few_shot_hs_file['post_test_res']['hidden_states'])):
            few_shot_hs.append(few_shot_hs_file['post_test_res']['hidden_states'][sample_index][args.injected_layer_num][0][-1])
    loss = linear_regression(zero_shot_hs, few_shot_hs, auto_encoder, epoch=50000)
    if args.num_epochs != 0:
        print("warning: --num_epochs is set, continual training?")
elif args.trained_autoencoder is not None:
    print(f"Loading trained autoencoder from {args.trained_autoencoder}")
    file = pickle.load(open(args.trained_autoencoder, "rb"))
    auto_encoder.load_state_dict(file["auto_encoder_state_dict"])
    if args.num_epochs != 0:
        print("warning: --num_epochs is set, continual training?")
print(auto_encoder)

if args.random_ablate_heads:
    total_head_numbers = ICL_model.config.num_attention_heads
    ablated_heads = {}
    for layer_num in range(ICL_model.config.num_hidden_layers):
        if layer_num not in args.ablated_heads:
            ablated_heads[layer_num] = []
        else:
            ablated_numbers = len(args.ablated_heads[layer_num])
            ablated_heads[layer_num] = random.sample(range(total_head_numbers), ablated_numbers)
    args.ablated_heads = ablated_heads

if "llama-3" in args.model_name.lower() or "llama3" in args.model_name.lower() or "s1" in args.model_name.lower():
    model_type = "llama-3"
elif "gpt2" in args.model_name.lower():
    model_type = "gpt2"
elif "qwen" in args.model_name.lower():
    model_type = "qwen"
elif "falcon3" in args.model_name.lower():
    model_type = "falcon3"
else:
    raise ValueError("Model type not supported. Please write a new model injection function for the new model type.")

############################# 
# Load the dataset
if args.ICL_dataset_index != -1:
    benchmark = StaICC.Normal(args.icl_demo_numbers)
    experimentor = benchmark[args.ICL_dataset_index]
    experimentor.prompt_former.replace_space_to_label()
    if args.instruction is not None:
        lmd.set_instruction_for_staicc_experimentor(experimentor, args.instruction)
    if args.symbolic_label:
        experimentor.prompt_former.change_label_space([" A", " B", " C", " D", " E", " F", " G", " H", " I", " J"][:len(experimentor.prompt_former.get_label_space())])
    prompts = None
    labels = None
    pure_inputs = None
    train_prompts = None
else:
    if args.ICL_dataset_name is None:
        raise ValueError("ICL dataset name must be provided when ICL_dataset_index is -1.")
    if args.ICL_dataset_name not in DATASET_NAME_TO_DATASET_CLASS:
        raise ValueError(f"Dataset {args.ICL_dataset_name} not supported. Please add it to DATASET_NAME_TO_DATASET_CLASS.")
    
    dataset_class = DATASET_NAME_TO_DATASET_CLASS[args.ICL_dataset_name]
    dataset = dataset_class(k=args.icl_demo_numbers)
    train_prompts, pure_inputs = dataset.make_train_samples(sample_number=2048)
    prompts, labels = dataset.make_ICL_samples()
    experimentor = None

############################# 
# Pre_test: very clean run on the original huggingface model
if not args.no_pre_test:
    if args.open_end_test:
        pre_test_res = test_model(
            model = ICL_model,
            tokenizer = ICL_tknz,
            experimentor = experimentor,
            prompts = prompts,
            labels = labels,
            ood_test = args.ood_test,
            id_test = args.id_test,
            wrong_demo_labels = args.random_label_word_test,
            ablate_last_label = args.ablate_last_label
        )
        print("Pre test result: ", pre_test_res)
    else:
        inference = partial(model_kernel.standard_ICL_inference_with_torch_Causal_LM, model = ICL_model, tokenizer = ICL_tknz, label_space = experimentor.get_label_space())
        pre_test_res = experimentor(inference)
        print("Pre test result: ", pre_test_res)
else:
    print("Skipping pre-test as per argument --no_pre_test.")
    pre_test_res = None

############################# 
# If pre_test_only is set, we only perform the pre-test and exit
if args.pre_test_only:
    print("Pre-test only mode. Exiting after pre-test.")
    path = args.save_path
    os.makedirs(path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = os.path.join(path, f"{args.model_name.split('/')[-1]}_{model_type}_{args.injected_layer_num}_{args.injected_rank}_{args.injected_activation}_{args.bias}_{timestamp}.txt")

    with open(filename, "w") as f:
        f.write(f"args: {args}\n")
        f.write(f"pre_test_res: {pre_test_res}\n")

        save_obj = {
            "args": args,
            "pre_test_res": pre_test_res,
        }

    pickle_filename = os.path.join(path, f"{args.model_name.split('/')[-1]}_{model_type}_{args.injected_layer_num}_{args.injected_rank}_{args.injected_activation}_{args.bias}_{timestamp}.pkl")
    with open(pickle_filename, "wb") as pf:
        pickle.dump(save_obj, pf)
    exit(0)

############################# 
# Make the injected model
if model_type == "llama-3":
    injected_model = Llama3_injected(
        llama3_model = ICL_model,
        auto_encoder = auto_encoder,
        injected_layer_num = args.injected_layer_num,
        hook = args.hook,
        output_hidden_states = args.output_hidden_states,
        output_attentions = args.output_attentions,
        only_last_token_hidden_states = True if not args.output_full_hidden_states else False,
        ablated_heads = args.ablated_heads,
        amplified_head = args.amplified_head,
        amplify_factor = args.amplify_factor
    )
elif model_type == "qwen":
    if args.amplified_head != {}:
        print("Warning: amplified_head is not supported currently for Qwen model. Ignoring amplified_head argument.")
    injected_model = Qwen2_injected(
        qwen2_model = ICL_model,
        auto_encoder = auto_encoder,
        injected_layer_num = args.injected_layer_num,
        hook = args.hook,
        output_hidden_states = args.output_hidden_states,
        output_attentions = args.output_attentions,
        only_last_token_hidden_states = True if not args.output_full_hidden_states else False,
        ablated_heads = args.ablated_heads,
        # amplified_head = args.amplified_head,
        # amplify_factor = args.amplify_factor
    )
else:
    raise ValueError("Model type not supported. Please write a new model injection function for the new model type.")

injected_model.gradient_on(train_part = args.train_part)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
wandb.init(
    project="Verb",
    name=current_time,
    config=args, 
    save_code=True
)


############################# 
# Train
trainer = Trainer(
    ICL_model = injected_model,
    ICL_tokenizer = ICL_tknz,
    experimentor = experimentor,
    train_dataset = train_prompts,
    pure_inputs = pure_inputs,
    training_parameters = {
        "train_sample_num" : 2048,
        "lr" : 1e-4,
        "optimizer" : torch.optim.Adam,
        "pseudo_batch_size" : 32, # We sum the gradients of the pseudo_batch_size samples and then update the model.
    },
    demonstration_number = args.icl_demo_numbers
)

max_acc = -1.0
min_loss = float('inf')
max_res_dict = None

for epoch in range(args.num_epochs):
    print(f"Epoch {epoch + 1}/{args.num_epochs}")
    loss = trainer.train(epochs=1)
    print(f"Loss: {loss}")
    if args.open_end_test:
        post_test_res = test_model(
            model = injected_model,
            tokenizer = ICL_tknz,
            experimentor = experimentor,
            prompts = prompts,
            labels = labels,
            ood_test = args.ood_test,
            id_test = args.id_test,
            wrong_demo_labels = args.random_label_word_test,
            ablate_last_label = args.ablate_last_label
        )
        current_acc = post_test_res['res']
        type_of_res = post_test_res['type']
        print("Post test result: ", post_test_res)
    else:
        inference = partial(standard_ICL_inference_with_injected, model = injected_model, tokenizer = ICL_tknz, label_space = experimentor.get_label_space())
        post_test_res = experimentor(inference)
        current_acc = post_test_res[0]["accuracy"]
        type_of_res == "acc"
        print("Post test result: ", post_test_res)
    wandb.log({
        "epoch": epoch + 1,
        "loss": loss,
        "accuracy": current_acc,
        "type_of_res": type_of_res
    })
    if type_of_res == "acc":
        if current_acc > max_acc:
            max_acc = current_acc
            max_res_dict = post_test_res
            print(f"New max accuracy: {max_acc}")
            wandb.log({"max_accuracy": max_acc})
    elif type_of_res == "loss":
        if loss < min_loss:
            min_loss = loss
            max_res_dict = post_test_res
            print(f"New min loss: {min_loss}")
            wandb.log({"min_loss": min_loss})

############################# 
# Test
if args.num_epochs == 0:
    loss = 0.0
    if args.open_end_test:
        max_res_dict = test_model(
            model = injected_model,
            tokenizer = ICL_tknz,
            experimentor = experimentor,
            prompts = prompts,
            labels = labels,
            ood_test = args.ood_test,
            id_test = args.id_test,
            wrong_demo_labels = args.random_label_word_test,
            ablate_last_label = args.ablate_last_label
        )
    else:
        inference = partial(standard_ICL_inference_with_injected, model = injected_model, tokenizer = ICL_tknz, label_space = experimentor.get_label_space())
        max_res_dict = experimentor(inference)

############################# 
# Save the results
path = args.save_path
os.makedirs(path, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
filename = os.path.join(path, f"{args.model_name.split('/')[-1]}_{model_type}_{args.injected_layer_num}_{args.injected_rank}_{args.injected_activation}_{args.bias}_{timestamp}.txt")

auto_encoder.cpu()

wandb.finish()

with open(filename, "w") as f:
    f.write(f"args: {args}\n")
    f.write(f"loss: {loss}\n")
    f.write(f"pre_test_res: {pre_test_res}\n")
    if args.hook:
        f.write(f"post_test_res: {max_res_dict['res']}\n")
    else:
        f.write(f"post_test_res: {max_res_dict}\n")

    save_obj = {
        "args": args,
        "loss": loss,
        "pre_test_res": pre_test_res,
        "post_test_res": max_res_dict,
        "auto_encoder_state_dict": auto_encoder.state_dict(),
    }

    pickle_filename = os.path.join(path, f"{args.model_name.split('/')[-1]}_{model_type}_{args.injected_layer_num}_{args.injected_rank}_{args.injected_activation}_{args.bias}_{timestamp}.pkl")
    with open(pickle_filename, "wb") as pf:
        pickle.dump(save_obj, pf)