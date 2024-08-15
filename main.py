from CutYourText.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
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