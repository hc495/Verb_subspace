import torch
import StaICC.util.functional as functional
import numpy as np
from tqdm import tqdm as tqdm
import random
import re


def standard_ICL_inference_with_injected(
    prompt: str,
    model: callable,
    tokenizer: callable,
    label_space: list[str],
    cache_empty: callable = torch.cuda.empty_cache(), # GPU cache empty function. Can be torch.cuda.empty_cache.
    calibration_function: callable = None, # standard calibration receives label_space_prob, full_vocab_prob, hidden_state, returns probabilities distribution aligned to the label_space
    return_hidden_state: bool = False,
    return_full_vocab_prob: bool = False
):
    with torch.no_grad():
        if cache_empty is not None:
            cache_empty()
        tknzd_data = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device) # flexable??
        result = model(tknzd_data, injected_token_index = len(tknzd_data[0]) - 1, output_hidden_states = True)
        full_vocab_prob = result['logits'][0][-1].detach().to(torch.float).cpu().numpy()
        tokenized_label_space = [tokenizer(label).input_ids[-1] for label in label_space] # The last token only
        label_space_logits = [full_vocab_prob[token] for token in tokenized_label_space]
        label_space_prob = functional.softmax(label_space_logits)
        del tknzd_data
        del result
        ret = label_space_prob
        if return_full_vocab_prob:
            if return_hidden_state:
                ret.append(full_vocab_prob)
            else:
                ret = (ret, full_vocab_prob)
        return ret
    

def __test_model_on_open_decoding(
    prompts: list[str],
    labels: list[str],
    model: callable,
    tokenizer: callable,
    task_vector: torch.Tensor = None,
    task_vector_layer: int = None
):
    # 判断labels是否均为单词（不含空格）
    if all(isinstance(label, str) and ' ' not in label.strip() for label in labels):
        res = __test_model_acc_on_open_decoding_classification(prompts, labels, model, tokenizer, task_vector, task_vector_layer)
        type = "acc"
    else:
        res = __test_model_loss_on_open_decoding(prompts, labels, model, tokenizer, task_vector, task_vector_layer)
        type = "loss"
    return {'type': type, 'res': res[0], 'hidden_states': res[1]}


def __test_model_acc_on_open_decoding_classification(
    prompts: list[str],
    labels: list[str],
    model: callable,
    tokenizer: callable,
    task_vector: torch.Tensor = None,
    task_vector_layer: int = None
):
    res = []
    hidden_states = []
    with torch.no_grad():
        for i in tqdm(range(len(prompts))):
            prompt = prompts[i]
            label = labels[i]
            tknzd_data = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            if hasattr(model, 'injected_layer_num'):
                result = model(tknzd_data, injected_token_index = len(tknzd_data[0]) - 1, task_vector = task_vector, task_vector_layer = task_vector_layer)
            else:
                result = model(tknzd_data, task_vector = task_vector, task_vector_layer = task_vector_layer)
            full_vocab_prob = result['logits'][0][-1].detach().to(torch.float).cpu().numpy()
            hidden_state = result.hidden_states # Notice: transfer to CPU and numpy in the inner model forward()
            decoded_output = np.argmax(full_vocab_prob)
            decoded_output = tokenizer.decode(decoded_output)
            if decoded_output == label:
                res.append(1)
            else:
                res.append(0)
            hidden_states.append(hidden_state)
    return np.mean(res), hidden_states


def __test_model_loss_on_open_decoding(
    prompts: list[str],
    labels: list[str],
    model: callable,
    tokenizer: callable,
    task_vector: torch.Tensor = None,
    task_vector_layer: int = None
):
    res = []
    hidden_states = []

    ## inputs may not contains the label part, but hidden states should be returned normally on the last forerunner token.
    if hasattr(model, 'only_last_token_hidden_states'):
        cached_only_last_token_hidden_states = model.only_last_token_hidden_states
        model.only_last_token_hidden_states = False
    else:
        cached_only_last_token_hidden_states = False

    with torch.no_grad():
        for i in tqdm(range(len(prompts))):
            prompt = prompts[i]
            label = labels[i]
            if prompt[-1] == ' ' or label[0] == ' ':
                full_input = prompt + label
            else:
                full_input = prompt + ' ' + label
            tknzd_data = tokenizer(full_input, return_tensors="pt").input_ids.to(model.device)
            tknzd_prompt = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            if hasattr(model, 'injected_layer_num') and not (model.injected_layer_num > model.config.num_hidden_layers or model.injected_layer_num < 0): 
                # Is the previous token blocked? (Fig.1 (C) in the paper)
                tknzd_label = tknzd_data[:, len(tknzd_prompt[0]) - 1:].clone()
            else:
                tknzd_label = tknzd_data.clone()
                tknzd_label[:, :len(tknzd_prompt[0]) - 1] = -100
            if hasattr(model, 'injected_layer_num') and not (model.injected_layer_num > model.config.num_hidden_layers or model.injected_layer_num < 0):
                result = model(tknzd_data, injected_token_index = len(tknzd_prompt[0]) - 1, labels = tknzd_label, task_vector = task_vector, task_vector_layer = task_vector_layer)
            else:
                result = model(tknzd_data, labels = tknzd_label, task_vector = task_vector, task_vector_layer = task_vector_layer)
            hidden_state = result.hidden_states # Notice: transfer to CPU and numpy in the inner model forward()
            if hidden_state is not None:
                if type(hidden_state) == list or type(hidden_state) == tuple:
                    hidden_state = list(hidden_state)
                    for layer in range(len(hidden_state)):
                        if cached_only_last_token_hidden_states:
                            hidden_state[layer] = hidden_state[layer][:, len(tknzd_prompt[0]) - 1:len(tknzd_prompt[0]), :]
                        else:
                            hidden_state[layer] = hidden_state[layer][:, :len(tknzd_prompt[0]), :]
                else:
                    for layer in range(len(hidden_state)):
                        if cached_only_last_token_hidden_states:
                            hidden_state[layer] = hidden_state['hidden_states'][layer][:, len(tknzd_prompt[0]) - 1:len(tknzd_prompt[0]), :]
                        else:
                            hidden_state[layer] = hidden_state['hidden_states'][layer][:, :len(tknzd_prompt[0]), :]
            loss = result['loss'].detach().to(torch.float).cpu()
            if loss.isnan() or loss.isinf():
                print(f"Loss is NaN or Inf for prompt: {prompt}, label: {label}")
                continue
            res.append(loss.item())
            hidden_states.append(hidden_state)

    if hasattr(model, 'only_last_token_hidden_states'):
        model.only_last_token_hidden_states = cached_only_last_token_hidden_states
    return np.mean(res), hidden_states


def test_model(
    model: callable,
    tokenizer: callable,
    experimentor = None,
    prompts = None,
    labels = None,
    ood_test = False, # If True, use the prompts and labels from the experimentor
    id_test = False, # If True, use the prompts and labels from the experimentor
    wrong_demo_labels = False,
    ablate_last_label = False,
    task_vector = None,
    task_vector_layer = None
):
    if experimentor is not None:
        return test_model_on_staicc_experimentor(model, tokenizer, experimentor, ood_test, id_test, wrong_demo_labels, ablate_last_label, task_vector, task_vector_layer)
    if ood_test or wrong_demo_labels or ablate_last_label or id_test:
        print("Warning: ood_test or wrong_demo_labels or ablate_last_label or id_test is True, but no experimentor is provided. Using prompts and labels from the arguments.")
    return __test_model_on_open_decoding(prompts, labels, model, tokenizer, task_vector, task_vector_layer)


def test_model_on_staicc_experimentor(
    model: callable,
    tokenizer: callable,
    experimentor: callable,
    ood_test: bool = False, # If True, use the prompts and labels from the experimentor
    id_test: bool = False, # If True, use the prompts and labels from the experimentor
    random_demo_labels: bool = False, # If True, use the wrong demo labels from the
    ablate_last_label: bool = False, # If True, ablate the last label in the prompts
    task_vector: torch.Tensor = None,
    task_vector_layer: int = None
):
    if id_test and ood_test:
        raise ValueError("Cannot perform both in-domain and out-of-domain tests at the same time.")
    if id_test:
        experimentor.set_in_domain_mode()
    if ood_test:
        experimentor.set_out_of_domain_mode()
    prompts = experimentor.prompt_set()
    if random_demo_labels:
        label_space = experimentor.get_label_space()
        label_pre_fix = experimentor.prompt_former._label_prefix
        new_prompts = []
        for prompt in prompts:
            pattern = re.escape(label_pre_fix.strip()) + r'\s*(\S+)'
            def replace_label(match):
                return label_pre_fix + random.choice(label_space)
            new_prompt, count = re.subn(pattern, replace_label, prompt)
            if count == 0:
                new_prompt = prompt
            new_prompts.append(new_prompt)
        prompts = new_prompts
    if ablate_last_label:
        label_pre_fix = experimentor.prompt_former._label_prefix
        new_prompts = []
        for prompt in prompts:
            pattern = re.escape(label_pre_fix.strip()) + r'\s*(\S+)'
            matches = list(re.finditer(pattern, prompt))
            if matches:
                last_match = matches[-1]
                start, end = last_match.span()
                new_prompt = prompt[:start] + label_pre_fix + ' ' + prompt[end:]
            else:
                new_prompt = prompt
            new_prompts.append(new_prompt)
        prompts = new_prompts
        print(prompts)
    labels = []
    for i in range(len(experimentor.triplet_dataset.test) * experimentor._repeat_times):
        label_index = experimentor.triplet_dataset.test.find_index_from_label(
            experimentor.triplet_dataset.test.get_label(i%len(experimentor.triplet_dataset.test))
        )
        labels.append(experimentor.get_label_space()[label_index])
    if ood_test:
        experimentor.reset_demonstration_sampler()
    return __test_model_on_open_decoding(prompts, labels, model, tokenizer, task_vector, task_vector_layer)


def ICL_inference_to_last_token_hidden_states(model, tokenizer, prompts): # [prompt] -> [layer][prompt][dimension]
    with torch.no_grad():
        ret = []
        hidden_states_in_layers = []
        for prompt in tqdm(prompts):
            torch.cuda.empty_cache()
            hidden_states_in_layer = []
            tknzd_data = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            result = model(tknzd_data, output_hidden_states = True)
            for layer in range(len(result.hidden_states)):
                hidden_states_in_layer.append(result.hidden_states[layer][-1][-1].detach().to(torch.float).cpu().numpy())
            hidden_states_in_layers.append(hidden_states_in_layer)
        for layer in range(len(hidden_states_in_layers[0])):
            layer_hidden_states = []
            for prompt in hidden_states_in_layers:
                layer_hidden_states.append(prompt[layer])
            ret.append(layer_hidden_states)
        return ret
    

def encoder_inference_to_feature(model, tokenizer, queries):
    with torch.no_grad():
        representations = []
        for query in tqdm(queries):
            tknzd_data = tokenizer(query, return_tensors="pt").input_ids.to(model.device)
            result = model(tknzd_data, output_hidden_states = True)
            representations.append(result.pooler_output[-1].detach().to(torch.float).cpu().numpy())
        return representations