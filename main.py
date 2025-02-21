from CutYourText.pipeline.feature_pineline import FeaturePipeline
from CutYourText.pipeline.training_pineline import TrainingPipeline
from CutYourText.logging import logger



STAGE_NAME = "Feature stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   feature_pineline = FeaturePipeline()
   feature_pineline.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n\n")
except Exception as e:
        logger.exception(e)
        raise e
     

STAGE_NAME = "Training stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   training_pineline = TrainingPipeline()
   training_pineline.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n\n")
except Exception as e:
        logger.exception(e)
        raise e   
     