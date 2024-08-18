from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
import torch
import os
from torch.optim import AdamW
from CutYourText.entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        
    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_bart = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_bart)
        
        # Load data
        dataset_dialogsum_pt = load_from_disk(self.config.data_path)

        # Define training arguments
        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir, 
            num_train_epochs=self.config.num_train_epochs, 
            warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size, 
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            weight_decay=self.config.weight_decay, 
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.eval_strategy, 
            eval_steps=self.config.eval_steps, 
            save_steps= self.config.save_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            fp16=self.config.fp16,
            save_total_limit=self.config.save_total_limit
        ) 

        # Define the optimizer using torch.optim.AdamW
        optimizer = AdamW(
            model_bart.parameters(),
            lr=3e-5,             # Fine-tuning learning rate for text summarization
            weight_decay=0.01,    # Standard weight decay
            eps=1e-8              # Epsilon to avoid division by zero in Adam
        )

        # Initialize Trainer with the custom optimizer
        trainer = Trainer(
            model=model_bart, 
            args=trainer_args,
            tokenizer=tokenizer, 
            data_collator=seq2seq_data_collator,
            train_dataset=dataset_dialogsum_pt["train"], 
            eval_dataset=dataset_dialogsum_pt["validation"],
            optimizers=(optimizer, None)  
        )
        
        trainer.train()

        # Save model
        model_bart.save_pretrained(os.path.join(self.config.root_dir, "bart-dialogsum-model"))
        # Save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))