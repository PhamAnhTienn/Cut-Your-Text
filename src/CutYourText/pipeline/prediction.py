from CutYourText.config.configuration import ConfigurationManager
from transformers import AutoTokenizer
from transformers import pipeline

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    def predict(self,text):
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        gen_kwargs = {
            "length_penalty": 1.0, 
            "num_beams": 4, 
            "max_length": 150, 
            "min_length": 30, 
            "early_stopping": True
        }

        pipe = pipeline("summarization", model=self.config.model_path,tokenizer=tokenizer)
        output = pipe(text, **gen_kwargs)[0]["summary_text"]

        return output