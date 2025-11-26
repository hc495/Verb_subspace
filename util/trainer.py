import torch
import random
from tqdm import tqdm
import math

class Trainer():
    def __init__(
        self, 
        ICL_model : torch.nn.Module,
        ICL_tokenizer,
        experimentor,
        train_dataset = None,
        pure_inputs = None,
        training_parameters = {
            "train_sample_num" : 2048,
            "lr" : 1e-4,
            "optimizer" : torch.optim.Adam,
            "pseudo_batch_size" : 16, # We sum the gradients of the pseudo_batch_size samples and then update the model.
        },
        demonstration_number = 0
    ):
        self.model = ICL_model
        self.tokenizer = ICL_tokenizer
        self.optimizer = training_parameters["optimizer"](
            filter(lambda p: p.requires_grad, self.model.auto_encoder.parameters()), 
            lr=training_parameters["lr"]
        )
        self.pseudo_batch_size = training_parameters["pseudo_batch_size"]
        self.train_sample_num = training_parameters["train_sample_num"]

        if train_dataset is None or pure_inputs is None:
            self.train_dataset, self.pure_inputs = self._make_test_samples_from_experimentor(
                experimentor = experimentor,
                sample_number = self.train_sample_num,
                sample_random_seed = 42,
                demonsration_number = demonstration_number
            )
        else:
            self.train_dataset = train_dataset
            self.pure_inputs = pure_inputs

    def _make_test_samples_from_experimentor(
        self, 
        experimentor,
        sample_number = 2048,
        sample_random_seed = 42,
        demonsration_number = 0
    ):
        original_set = experimentor.calibration_set()
        prompt_writter = experimentor.prompt_former
    
        ## Sample k+1 inputs
        now_random_seed = sample_random_seed
        sampled_indics = []
        for i in range(sample_number):
            random.seed(now_random_seed)
            sampled_indics.append(random.sample(range(len(original_set)), demonsration_number + 1))
            now_random_seed += 1
        
        ## Find label sets
        labels = []
        for samples in sampled_indics:
            label_word = experimentor.prompt_former.get_label_space()[original_set.find_index_from_label(original_set[samples[-1]][1])]
            labels.append(label_word)
        
        ## Form the prompts
        pure_inputs = []
        for i in range(sample_number):
            demo_lines = []
            for index in range(len(sampled_indics[i]) - 1):
                demo_lines.append(original_set[sampled_indics[i][index]])
            query_line = original_set[sampled_indics[i][-1]][0]
            prompt = prompt_writter.write_prompt_from_dataline(demo_lines, query_line)
            if prompt_writter._label_prefix[-1] == ' ':
                prompt = prompt[:-1]
            pure_inputs.append(prompt)
        
        prompts = []
        for i in range(sample_number):
            if labels[i][0] == ' ':
                prompts.append(pure_inputs[i] + labels[i])
            else:
                prompts.append(pure_inputs[i] + ' ' + labels[i])

        return prompts, pure_inputs

    def train(self, epochs=4):
        self.model.train()
        self.optimizer.zero_grad()
        for epoch in range(epochs):
            epoch_loss = 0
            with tqdm(range(math.ceil(self.train_sample_num / self.pseudo_batch_size)), desc=f"Epoch {epoch+1}") as pbar:
                for i in pbar:
                    batch_start = i * self.pseudo_batch_size
                    batch_end = min((i + 1) * self.pseudo_batch_size, self.train_sample_num)
                    pseudo_batch = self.train_dataset[batch_start:batch_end]
                    pseudo_pure_input_ids_batch = self.pure_inputs[batch_start:batch_end]
                    input_ids = [self.tokenizer.encode(sample, return_tensors="pt").to(self.model.device) for sample in pseudo_batch]
                    pure_input_ids = [self.tokenizer.encode(sample, return_tensors="pt").to(self.model.device) for sample in pseudo_pure_input_ids_batch]
                    
                    filling_token_length_in_labels = []
                    for j in range(len(input_ids)):
                        filling_token_length_in_labels.append(len(pure_input_ids[j][0]))

                    labels = []
                    for j in range(len(input_ids)):
                        labels.append(input_ids[j][0][filling_token_length_in_labels[j]-1:].clone())

                    filter_positions = []
                    for j in range(len(input_ids)):
                        filter_positions.append(len(pure_input_ids[j][0]) - 1)

                    # Forward pass
                    for j in range(len(input_ids)):
                        outputs = self.model(
                            input_ids = input_ids[j],
                            injected_token_index = filter_positions[j],
                            labels = labels[j],
                        )
                        if j == 0:
                            loss = outputs.loss
                        else:
                            loss += outputs.loss
                    loss /= len(input_ids)             
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    epoch_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item()})
        return loss.item()

    # def train(self):
    #     self.model.train()
    #     self.optimizer.zero_grad()
    #     for epoch in range(self.epoch):
    #         for i in tqdm(range(math.ceil(self.train_sample_num / self.pseudo_batch_size))):
    #             batch_start = i * self.pseudo_batch_size
    #             batch_end = min((i + 1) * self.pseudo_batch_size, self.train_sample_num)
    #             pseudo_batch = self.train_dataset[batch_start:batch_end]
    #             pseudo_pure_input_ids_batch = self.pure_inputs[batch_start:batch_end]
    #             input_ids = [self.tokenizer.encode(sample, return_tensors="pt").to(self.model.device) for sample in pseudo_batch]
    #             pure_input_ids = [self.tokenizer.encode(sample, return_tensors="pt").to(self.model.device) for sample in pseudo_pure_input_ids_batch]
                
    #             pad_token_length_in_labels = []
    #             for j in range(len(input_ids)):
    #                 pad_token_length_in_labels.append(len(pure_input_ids[j][0]))

    #             labels = []
    #             for j in range(len(input_ids)):
    #                 labels.append(input_ids[j][0].clone())
    #                 labels[j][:pad_token_length_in_labels[j]] = -100
                
    #             filter_positions = []
    #             for j in range(len(input_ids)):
    #                 filter_positions.append([len(pure_input_ids[j][0]) - 1])

    #             # Forward pass
    #             for j in range(len(input_ids)):
    #                 outputs = self.model(
    #                     input_ids = input_ids[j],
    #                     injected_token_index = filter_positions[j],
    #                     labels = labels[j],
    #                 )
    #                 if j == 0:
    #                     loss = outputs.loss
    #                 else:
    #                     loss += outputs.loss
    #             loss /= len(input_ids)             
    #             loss.backward()
    #             self.optimizer.step()
    #             self.optimizer.zero_grad()