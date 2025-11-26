from StaICC.util.dataset_interface import demonstration_sampler
import random
import datasets

class capital():
    # https://github.com/hc495/country-capitals/blob/master/data/country-list.csv
    def __init__(self, 
        k = 0,
        path = "experiment_matrials/datasets/country-list.csv",
        instruction = None
    ):
        self.dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]  # Skip the header
            for line in lines:
                self.dataset.append(line.strip().split(','))
        self.dataset = [[item.replace('"', '') for item in row] for row in self.dataset]
        
        random.seed(42)
        random.shuffle(self.dataset)
        total = len(self.dataset)
        train_end = int(total * 0.5)
        val_end = train_end + int(total * 0.3)
        self.calibration = self.dataset[:train_end]
        self.demonstration = self.dataset[train_end:val_end]
        self.test = self.dataset[val_end:]
        self.instruction = instruction
        self.demonstration_sampler = demonstration_sampler(
            k = k, 
            demonstration_set_size = len(self.demonstration),
            query_numbers = len(self.test)
        )
    
    def make_train_samples(
        self, 
        sample_number = 2048,
        sample_random_seed = 42
    ):
        now_random_seed = sample_random_seed
        sampled_indics = []
        for i in range(sample_number):
            random.seed(now_random_seed)
            sampled_indics.append(random.sample(range(len(self.calibration)), 1))
            now_random_seed += 1
        
        labels = []
        for samples in sampled_indics:
            label_word = self.calibration[samples[0]][1]
            labels.append(label_word)
        
        pure_inputs = []
        prompts = []
        for i in range(sample_number):
            if self.instruction is not None:
                inputs = self.instruction + "country: " + self.calibration[sampled_indics[i][0]][0] + ", label:"
            else:
                inputs = "country: " + self.calibration[sampled_indics[i][0]][0] + ", label:"
            pure_inputs.append(inputs)
            prompts.append(inputs + ' ' + labels[i])
        
        return prompts, pure_inputs

    def make_ICL_samples(
        self
    ):
        inputs = []
        labels = []
        for i in range(len(self.demonstration_sampler)):
            sampled = self.demonstration_sampler[i]
            temp_prompts = ''
            if self.instruction is not None:
                temp_prompts += self.instruction
            for j in range(len(sampled)):
                temp_prompts += 'country: ' + self.demonstration[sampled[j]][0] + ', label: ' + self.demonstration[sampled[j]][1] + '\n'
            temp_prompts += 'country: ' + self.test[i][0] + ', label:'
            inputs.append(temp_prompts)
            labels.append(self.test[i][1])
        return inputs, labels

class famous_people_prof():
    # https://medialab.github.io/bhht-datascape/
    def __init__(self, 
        k = 0,
        path = "experiment_matrials/datasets/cross-verified-database.csv",
        instruction = None
    ):
        self.column = 23
        self.dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:20000]  # Skip the header
            for line in lines:
                self.dataset.append(line.strip().split(','))
        
        # Remove rows where the 23rd column is empty
        self.dataset = [row for row in self.dataset if len(row) > self.column and row[self.column].strip() != '']

        random.seed(42)
        random.shuffle(self.dataset)
        total = len(self.dataset)
        train_end = int(total * 0.5)
        val_end = train_end + int(total * 0.3)
        self.calibration = self.dataset[:train_end]
        self.demonstration = self.dataset[train_end:val_end]
        self.test = self.dataset[val_end:]
        self.instruction = instruction
        self.demonstration_sampler = demonstration_sampler(
            k = k, 
            demonstration_set_size = len(self.demonstration),
            query_numbers = len(self.test)
        )

    def line_to_space(self, string):
        return string.replace('_', ' ')
    
    def make_train_samples(
        self, 
        sample_number = 2048,
        sample_random_seed = 42
    ):
        now_random_seed = sample_random_seed
        sampled_indics = []
        for i in range(sample_number):
            random.seed(now_random_seed)
            sampled_indics.append(random.sample(range(len(self.calibration)), 1))
            now_random_seed += 1
        
        labels = []
        for samples in sampled_indics:
            label_word = self.calibration[samples[0]][self.column]
            labels.append(label_word)
        
        pure_inputs = []
        prompts = []
        for i in range(sample_number):
            if self.instruction is not None:
                inputs = self.instruction + "name: " + self.line_to_space(self.calibration[sampled_indics[i][0]][12]) + ", label:"
            else:
                inputs = "name: " + self.line_to_space(self.calibration[sampled_indics[i][0]][12]) + ", label:"
            pure_inputs.append(inputs)
            prompts.append(inputs + ' ' + labels[i])
        
        return prompts, pure_inputs

    def make_ICL_samples(
        self
    ):
        inputs = []
        labels = []
        for i in range(len(self.demonstration_sampler)):
            sampled = self.demonstration_sampler[i]
            temp_prompts = ''
            if self.instruction is not None:
                temp_prompts += self.instruction
            for j in range(len(sampled)):
                temp_prompts += 'name: ' + self.line_to_space(self.demonstration[sampled[j]][12]) + ', label: ' + self.demonstration[sampled[j]][self.column] + '\n'
            temp_prompts += 'name: ' + self.line_to_space(self.test[i][12]) + ', label:'
            inputs.append(temp_prompts)
            labels.append(self.test[i][self.column])
        return inputs, labels
    
class translation():
    # https://huggingface.co/datasets/Helsinki-NLP/opus-100?library=datasets
    def __init__(self, 
        k = 0,
        path = None,
        instruction = None
    ):
        self.column = 23
        self.dataset = []
        self.huggingface_name = "Helsinki-NLP/opus-100"
        dataset = datasets.load_dataset(self.huggingface_name, 'en-zh', split='train')

        for item in dataset['translation'][0:4096]:
            self.dataset.append([item['en'], item['zh']])

        random.seed(42)
        random.shuffle(self.dataset)
        total = len(self.dataset)
        train_end = int(total * 0.5)
        val_end = train_end + int(total * 0.3)
        self.calibration = self.dataset[:train_end]
        self.demonstration = self.dataset[train_end:val_end]
        self.test = self.dataset[val_end:]
        self.instruction = instruction
        self.demonstration_sampler = demonstration_sampler(
            k = k, 
            demonstration_set_size = len(self.demonstration),
            query_numbers = len(self.test)
        )

    def make_train_samples(
        self, 
        sample_number = 2048,
        sample_random_seed = 42
    ):
        now_random_seed = sample_random_seed
        sampled_indics = []
        for i in range(sample_number):
            random.seed(now_random_seed)
            sampled_indics.append(random.sample(range(len(self.calibration)), 1))
            now_random_seed += 1
        
        labels = []
        for samples in sampled_indics:
            label_word = self.calibration[samples[0]][1]
            labels.append(label_word)
        
        pure_inputs = []
        prompts = []
        for i in range(sample_number):
            if self.instruction is not None:
                inputs = self.instruction + "sentence: " + self.calibration[sampled_indics[i][0]][0] + ", translation:"
            else:
                inputs = "sentence: " + self.calibration[sampled_indics[i][0]][0] + ", translation:"
            pure_inputs.append(inputs)
            prompts.append(inputs + ' ' + labels[i])
        
        return prompts, pure_inputs

    def make_ICL_samples(
        self
    ):
        inputs = []
        labels = []
        for i in range(len(self.demonstration_sampler)):
            sampled = self.demonstration_sampler[i]
            temp_prompts = ''
            if self.instruction is not None:
                temp_prompts += self.instruction
            for j in range(len(sampled)):
                temp_prompts += 'sentence: ' + self.demonstration[sampled[j]][0] + ', translation: ' + self.demonstration[sampled[j]][1] + '\n'
            temp_prompts += 'sentence: ' + self.test[i][0] + ', translation:'
            inputs.append(temp_prompts)
            labels.append(self.test[i][1])
        return inputs, labels
