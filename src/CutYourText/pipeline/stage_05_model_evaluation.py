from CutYourText.config.configuration import ConfigurationManager
from CutYourText.conponents.model_evaluation import ModelEvaluation
from CutYourText.logging import logger

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        try:
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            model_evaluation.evaluate()
        except Exception as e:
            raise e