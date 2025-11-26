from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig
import torch
import random
from StaICC.util import experimentor

def load_ICL_model(name: str, device: str = "cuda", huggingface_token = None, quantized = False, forcedownload = False, revision = None):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    ) if quantized else None
    if huggingface_token is not None:
        model = AutoModelForCausalLM.from_pretrained(name, token = huggingface_token, quantization_config = quantization_config, force_download = forcedownload, revision = revision, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(name, quantization_config = quantization_config, force_download = forcedownload, revision = revision, trust_remote_code=True)
    if not quantized:
        model.to(device)
    model.eval()
    if huggingface_token is not None:
        tokenizer = AutoTokenizer.from_pretrained(name, token = huggingface_token, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    return model, tokenizer

def find_tokenized_label_word_old(tokenizer, experimentor, prompt, pythia = False):
    tokenized_prompt_input_ids = tokenizer(prompt)['input_ids']
    fore_runner_loca = []
    label_words_loca = []
    if pythia:
        divider = experimentor.prompt_former._label_prefix[:-1]
    else:
        divider = ' ' + experimentor.prompt_former._label_prefix[:-1]
    tokenized_divider = tokenizer(divider)['input_ids']
    tokenized_divider = tokenized_divider[-2:]
    for i in range(len(tokenized_prompt_input_ids)):
        if tokenized_prompt_input_ids[i:i + len(tokenized_divider)] == tokenized_divider:
            fore_runner_loca.append(i + 1)
            label_words_loca.append(i + 2)
    return fore_runner_loca, label_words_loca

def find_tokenized_label_word(tokenizer, experimentor, prompt, pythia = False):
    tokenized_prompt_input_ids = tokenizer(prompt)['input_ids']
    fore_runner_loca = []
    label_words_loca = []
    if pythia:
        divider = experimentor.prompt_former._label_prefix[:-1]
    else:
        divider = ' ' + experimentor.prompt_former._label_prefix[:-1]
    tokenized_divider = tokenizer(divider)['input_ids']
    tokenized_divider = tokenized_divider[-2:]
    for i in range(len(tokenized_prompt_input_ids)):
        if tokenized_prompt_input_ids[i:i + len(tokenized_divider)] == tokenized_divider:
            fore_runner_loca.append(i + 1)
            label_words_loca.append(i + 2)
    return label_words_loca[:-1]

def make_test_samples_from_experimentor(
    experimentor,
    k,
    sample_number = 64,
    sample_random_seed = 42
):
    original_set = experimentor.calibration_set()
    prompt_writter = experimentor.prompt_former

    ## Sample k+1 inputs
    random.seed(sample_random_seed)
    sampled_indics = []
    for i in range(sample_number):
        sampled_indics.append(random.sample(range(len(original_set)), (k+1)))
    
    ## Find label sets
    labels = []
    for samples in sampled_indics:
        label = []
        for index in samples:
            label_word = original_set[index][1]
            label_index = prompt_writter._label_space.index(label_word)
            label.append(label_index)
        labels.append(label)
    
    ## Form the prompts
    prompts = []
    for i in range(sample_number):
        demo_lines = []
        for index in range(len(sampled_indics[i]) - 1):
            demo_lines.append(original_set[index])
        query_line = original_set[sampled_indics[i][-1]][0]
        prompt = prompt_writter.write_prompt_from_dataline(demo_lines, query_line)
        if prompt_writter._label_prefix[-1] == ' ':
            prompt = prompt[:-1]
        prompts.append(prompt)
    
    return prompts, labels

def transfer_label_indexs_to_true_and_false(labels):
    res = []
    for labelline in labels:
        line = []
        for i in range(len(labelline) - 1):
            if labelline[i] == labelline[-1]:
                line.append(True)
            else:
                line.append(False)
        res.append(line)
    return res

def reinitialize_experimentor_to_divide(benchmark, experimentor_index, divide):
    new_experimentor = experimentor.single_experimentor(
        original_dataset = benchmark._original_data[experimentor_index], 
        k=benchmark[experimentor_index]._k, 
        metrics=benchmark.metrics, 
        dividing=divide,
    )
    return new_experimentor

def load_data_from_StaICC_experimentor(experimentor):
    _queries = experimentor.test_set()
    prompts = experimentor.prompt_set()[:len(_queries)]
    pure_inputs = []
    
    for i in range(len(prompts)):
        pure_inputs.append(prompts[i])
        prompts[i] = prompts[i] + experimentor.prompt_former._label_space[_queries._label_space.index(_queries[i][1])]

    return prompts, pure_inputs

def load_prompts_and_queries_from_StaICC_experimentor(experimentor, prompt_cut = "none", target_label_correction = True):
    _queries = experimentor.test_set()
    prompts = experimentor.prompt_set()
    queries = []
    for i in range(len(_queries) * 2):
        queries.append(_queries[i%len(_queries)][0][0])
    cut_amount = -1
    if prompt_cut == "none":
        cut_amount = -1
    elif prompt_cut == "label_words":
        cut_amount = -1
        for i in range(len(prompts)):
            if target_label_correction:
                prompts[i] = prompts[i] + experimentor.prompt_former._label_space[_queries._label_space.index(_queries[i][1])] + ' '
            else:
                prompts[i] = prompts[i] + experimentor.prompt_former._label_space[(_queries._label_space.index(_queries[i][1]) + 1) % len(_queries._label_space)] + ' '
    elif prompt_cut == "last_sentence_token":
        label_prefix_length = len(experimentor.prompt_former._label_prefix)
        cut_amount = -label_prefix_length - 1
    
    for i in range(len(prompts)):
        prompts[i] = prompts[i][:cut_amount]
    print(len(prompts), len(queries))
    return prompts, queries

def load_encode_model(name: str, device: str = "cuda", huggingface_token = None):
    if huggingface_token is not None:
        model = AutoModel.from_pretrained(name, token = huggingface_token)
    else:
        model = AutoModel.from_pretrained(name)
    model.to(device)
    model.eval()
    if huggingface_token is not None:
        tokenizer = AutoTokenizer.from_pretrained(name, token = huggingface_token)
    else:
        tokenizer = AutoTokenizer.from_pretrained(name)
    return model, tokenizer

def set_instruction_for_staicc_experimentor(experimentor, instruction):
    if instruction is None:
        return
    experimentor.prompt_former.change_instruction(instruction)