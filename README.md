# Mechanism of Task-oriented Information Removal in In-context Learning

<p align="center">
  <a href="https://www.hakaze-c.com/">Hakaze Cho</a>, et al.
  <br>
  <a href="https://arxiv.org/abs/2509.21012">arXiv</a> •
  <a href="https://openreview.net/forum?id=VAv1rrPR1A">OpenReview</a>
  <br>
  <br>
  <a href="https://openreview.net/forum?id=VAv1rrPR1A"><img src="https://img.shields.io/badge/ICLR_2026-Accepted-blue?link=https%3A%2F%2Fopenreview.net%2Fforum%3Fid%3DxizpnYNvQq"></a>
  <a href="https://arxiv.org/abs/2509.21012"><img alt="Static Badge" src="https://img.shields.io/badge/arXiv-2509.21012-red?style=flat&link=https%3A%2F%2Farxiv.org%2Fabs%2F2509.21012"></a>
</p>


**This repo contains the official code for the following paper accepted at ICLR 2026:**

> Hakaze Cho, et al. **"Mechanism of Task-oriented Information Removal in In-context Learning."**

Implemented by [Hakaze Cho](https://www.hakaze-c.com/), the primary contributor of the paper.

## Overview

### Abstract

In-context Learning (ICL) is an emerging few-shot learning paradigm based on modern Language Models (LMs), yet its inner mechanism remains unclear. In this paper, we investigate the mechanism through a novel perspective of information removal. Specifically, we demonstrate that in the zero-shot scenario, LMs encode queries into non-selective representations in hidden states containing information for all possible tasks, leading to arbitrary outputs without focusing on the intended task, resulting in near-zero accuracy. Meanwhile, we find that selectively removing specific information from hidden states by a low-rank filter effectively steers LMs toward the intended task. Building on these findings, by measuring the hidden states on carefully designed metrics, we observe that few-shot ICL effectively simulates such task-oriented information removal processes, selectively removing the redundant information from entangled non-selective representations, and improving the output based on the demonstrations, which constitutes a key mechanism underlying ICL. Moreover, we identify essential attention heads inducing the removal operation, termed Denoising Heads, which enables the ablation experiments blocking the information removal operation from the inference, where the ICL accuracy significantly degrades, especially when the correct label is absent from the few-shot demonstrations, confirming both the critical role of the information removal mechanism and denoising heads.

### Summary figure

<p align="center">
<img src="https://s2.loli.net/2025/12/12/cM6K3QukE2FqJDh.png" width="80%" />
</p>

(A) A zero-shot query is encoded into a non-selective semantic representation containing all possible label information on various subspaces, making the output arbitrary among these labels. (B) Demonstrations help LM filter the label information, saving only the task-related one on the specific subspace (termed Task-Verbalization Subspace (TVS)), leading to the task-specific output. (C) We explicitly find the TVS by injecting a low-rank filter into the residual stream of zero-shot inputs, and train only the filter to drive the final outputs towards the ground-truth labels.

## Setup

### 0. Requirement

1. A GPU with more than 48GB VRAM and CUDA (Ver. `12.4` recommended) are strongly required to run all the experiments.
2. A local strorage with more than `1TB` free space is recommended.
3. Network connection to `huggingface` is needed to download the pre-trained model. And a `huggingface` user token with access to the [`Llama Family`](https://huggingface.co/meta-llama/Llama-2-7b) model is recommended to run a part of the experiments.
4. `Anaconda` or `Miniconda` is needed.

### 1. Clone this repo

```bash
git clone https://github.com/hc495/Verb_subspace.git
cd Verb_subspace
```

### 2. Environment Installation

```bash
conda env create -f environment.yaml
conda activate verb_subspace
```

## Major Experiment code

The file `main_experiments.py` contains the major experiment code to reproduce the results in the paper. You can run different experiments by changing the arguments.

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_name` | str | Required | Path to the pretrained model |
| `--ICL_dataset_index` | int | -1 | ICL dataset index, defined in StaICC. If -1, use ICL_dataset_name instead. |
| `--ICL_dataset_name` | str | None | ICL dataset name (only used when ICL_dataset_index is -1). Selection: 'capital', 'profession', 'translation' |
| `--huggingface_token` | str | None | Huggingface token for model access. Empty to use os.environ['HF_TOKEN'] or no token. |
| `--residual` | bool | False | Use residual connection in the injected filter |
| `--quantized` | bool | False | Use quantized model |
| `--injected_rank` | int | 16 | Rank of the injected filter |
| `--num_epochs` | int | 4 | Number of training epochs |
| `--injected_activation` | str | "none" | Activation function for the injected filter |
| `--injected_layer_num` | int | 0 | Layer number to inject the filter into |
| `--bias` | str | "both" | Bias for the injected filter (encoder, decoder, both, none) |
| `--save_path` | str | "logs" | Path to save the logs |
| `--open_end_test` | bool | False | Whether to perform open-end test |
| `--symbolic_label` | bool | False | Whether to use symbolic label |
| `--trained_autoencoder` | str | None | Path to a trained autoencoder to load |
| `--train_part` | str | "both" | Train on encoder, decoder, none, or both |
| `--icl_demo_numbers` | int | 0 | Number of ICL demos to use for training and testing |
| `--ood_test` | bool | False | Whether to perform OOD test. Only works for StaICC datasets. |
| `--id_test` | bool | False | Whether to perform in-domain test. Only works for StaICC datasets. |
| `--random_label_word_test` | bool | False | Whether to perform random label (noisy label) test. Only works for StaICC datasets. |
| `--hook` | bool | False | Enable hook mode |
| `--pre_test_only` | bool | False | Whether to only perform pre-test without training |
| `--no_pre_test` | bool | False | Whether to skip pre-test and directly train the model |
| `--output_hidden_states` | bool | False | Whether to output hidden states from the model |
| `--output_full_hidden_states` | bool | False | Whether to output full hidden states from the model |
| `--output_attentions` | bool | False | Whether to output attentions from the model |
| `--instruction` | str | None | Instruction to use for the model (if applicable). Typically space needed in the end. |
| `--ablated_heads` | dict | {} | Heads to ablate in the model. Format: {'layer_num': [head1, head2, ...]} |
| `--random_ablate_heads` | bool | False | Whether to randomly ablate heads in the model, with the same layer-wise amount as specified in --ablated_heads. |
| `--ablate_last_label` | bool | False | Whether to ablate the last label in the prompts. Only works for StaICC datasets. |
| `--estimate_filter` | bool | False | Whether to estimate the filter based on the model's hidden states. Will cover the trained_autoencoder. |
| `--saved_zero_shot_hs` | str | None | 0-shot hidden states for the filter estimation |
| `--saved_few_shot_hs` | str | None | few-shot hidden states for the filter estimation |
| `--amplify_factor` | float | 0.1 | Amplification factor for the amplified_head |
| `--amplified_head` | dict | {} | Head to amplify in the model. Format: {'layer_num': head_num} |

## Repeat Experiments

The folder `SOP` contains the scripts to repeat all the major experiments in the paper. Move the scripts to the root folder and run them.

In detail:

| Num | File Name | Experiment Description | Result in the Paper | Requirement |
|:---:|-----------|------------------------|---------------------| ------------- |
|  1  | `Exp_1_0_low_rank_filter.sh` | Train the explicit TVP and test the evaluation accuracy. | Fig. 3 | - |
|  2  | `Exp_1_1_symbolic.sh` | Train only one part of TVP (encoder/decoder) and test the evaluation accuracy with symbolic labels. | Table 1 | Experiment 1 |
|  3  | `Exp_1_2_captial.sh` | Train the explicit TVP on the capital dataset. | Fig. 12 | - |
|  4  | `Exp_1_3_profession.sh` | Train the explicit TVP on the profession dataset. | Fig. 13 | - |
|  5  | `Exp_1_4_encoding_magnitude.sh` | Analyze the encoding magnitude on various layers. Taken from https://github.com/hc495/ICL_Circuit, also my work so that no IP issues arise. | Fig. 3 | - |
|  6  | `Exp_2_0_hidden_state_geo.sh` | Collect the hidden states on various k and l. | Fig. 4 | - |
|  7  | `Exp_2_1_hidden_state_instruction.sh` | Collect the hidden states with instructions. | Fig. 5 | - |
|  8  | `Exp_2_2_k_8_condition.sh` | Collect the hidden states under the unseen and random label settings. | Fig. 6 | - |
|  9  | `Exp_3_0_head_ablation.sh` | Ablate each heads and re-evaluate the 2 metrics, also accuracy. | Fig. 7 | - |
| 10  | `Exp_3_1_induction_magnitude.ipynb` | Calculate the induction head magnitude. Taken from https://github.com/hc495/ICL_Circuit, also my work so that no IP issues arise. | Fig. 9 | - |
| 11  | `Exp_3_2_denoising_head_ablation_acc.sh` | Ablate the all the denoising heads and test the evaluation accuracy under various settings. | Table 2 | Experiment 9, Visualization 3 |
| 12  | `Exp_3_3_denoising_head_amp.sh` | Amplify the denoising heads and test the evaluation accuracy under various settings. | Fig. 14 | Experiment 9, Visualization 3 |

## Result Visualization

The folder `main_visualization` contains the code to visualize the results. You should redirect the path in these ipynbs to your saved log path before running them.

| Num | File Name | Visualization Description | Result in the Paper | Requirement |
|:---:|-----------|---------------------------|---------------------| ------------- |
|  1  | `explicit_tvp_acc.ipynb` | Visualize the explicit TVP accuracy results. | Fig. 3, Fig. 25 | Experiment 1, 4 |
|  2  | `eccen_and_cov.ipynb` | Visualize the eccentricity and cov flux results. | Fig. 4, Fig. 5, Fig. 6 | Experiment 1, 6, 7, 8 |
|  3  | `head_ablation_visualization.ipynb` | Visualize the head ablation and denoising head results. Also, find the denosing heads. | Fig. 7 | Experiment 1, 9 |

## Citation

If you find this work useful for your research, please cite [our paper](https://openreview.net/forum?id=VAv1rrPR1A):

```bibtex
@inproceedings{cho2026mechanism,
  title={Mechanism of Task-oriented Information Removal in In-context Learning},
  author={Hakaze Cho and Haolin Yang and Gouki Minegishi and Naoya Inoue},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=VAv1rrPR1A}
}
```