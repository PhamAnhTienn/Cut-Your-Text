from CutYourText.config.configuration import ConfigurationManager
from CutYourText.conponents.model_trainer import ModelTrainer, ModelEvaluation
from CutYourText.logging import logger

class TrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        try:
            config = ConfigurationManager()
            
            model_trainer_config = config.get_model_trainer_config()
            model_evaluation_config = config.get_model_evaluation_config()
            
            model_trainer = ModelTrainer(config=model_trainer_config)
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            
            model_trainer.train()
            model_evaluation.evaluate()
        except Exception as e:
            raise e