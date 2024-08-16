from CutYourText.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from CutYourText.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from CutYourText.logging import logger



STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n\n")
except Exception as e:
        logger.exception(e)
        raise e
     

STAGE_NAME = "Data Validation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_validation = DataValidationTrainingPipeline()
   data_validation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n\n")
except Exception as e:
        logger.exception(e)
        raise e   
