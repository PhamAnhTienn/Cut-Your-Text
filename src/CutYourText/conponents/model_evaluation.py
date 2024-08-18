from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from datasets import load_from_disk
from evaluate import load
import torch
import pandas as pd
from tqdm import tqdm
from CutYourText.entity import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def generate_batch_sized_chunks(self,list_of_elements, batch_size):
        """split the dataset into smaller batches that we can process simultaneously
        Yield successive batch-sized chunks from list_of_elements."""
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    def calculate_metric_on_test_ds(self,dataset, metric, model, tokenizer, 
                                    batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu", 
                                    column_text="article", column_summary="highlights"):
        generation_config = GenerationConfig(
            max_length=150,
            min_length=30,
            early_stopping=True,
            num_beams=4,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
        )
        
        article_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
        target_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))

        for article_batch, target_batch in tqdm(zip(article_batches, target_batches), total=len(article_batches)):
            inputs = tokenizer(article_batch, max_length=1024, truncation=True, padding="max_length", return_tensors="pt")
            
            summaries = model.generate(input_ids=inputs["input_ids"].to(device), 
                                       attention_mask=inputs["attention_mask"].to(device), 
                                       generation_config=generation_config)
            
            decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in summaries]      
            
            decoded_summaries = [d.replace("", " ") for d in decoded_summaries]
            
            metric.add_batch(predictions=decoded_summaries, references=target_batch)
  
        score = metric.compute()
        return score

    def evaluate(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model_bart = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)
       
        #loading data 
        dataset_dialogsum_pt = load_from_disk(self.config.data_path)

        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        rouge_metric = load("rouge")

        score = self.calculate_metric_on_test_ds(
            dataset_dialogsum_pt['test'][0:10], rouge_metric, model_bart, tokenizer, 
            batch_size = 2, column_text = 'dialogue', column_summary= 'summary'
        )

        rouge_dict = dict((rn, score[rn]) for rn in rouge_names)

        df = pd.DataFrame(rouge_dict, index = ['bart with early stopping'] )
        df.to_csv(self.config.metric_file_name, index=False)