import os
from CutYourText.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset
from CutYourText.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):
        # Tokenize the input dialogue
        input_encodings = self.tokenizer(example_batch['dialogue'], max_length=1024, truncation=True)
        
        # Tokenize the summary as target
        target_encodings = self.tokenizer(text_target=example_batch['summary'], max_length=128, truncation=True)
        
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }

    def convert(self):
        dataset_dialogsum = load_dataset('csv', data_files=self.config.data_path)
        dataset_dialogsum_pt = dataset_dialogsum.map(self.convert_examples_to_features, batched=True)
        dataset_dialogsum_pt.save_to_disk(os.path.join(self.config.root_dir, "dialogsum_dataset"))