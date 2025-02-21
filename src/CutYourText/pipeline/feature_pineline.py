from CutYourText.config.configuration import ConfigurationManager
from CutYourText.conponents.data_ingestion import DataIngestion, DataValidation, DataTransformation
from CutYourText.logging import logger


class FeaturePipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            
            data_ingestion_config = config.get_data_ingestion_config()
            data_validation_config = config.get_data_validation_config()
            data_transformation_config = config.get_data_transformation_config()
            
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_validation = DataValidation(config=data_validation_config)
            data_transformation = DataTransformation(config=data_transformation_config)
            
            data_ingestion.download_file()
            data_ingestion.extract_zip_file()
            data_validation.validate_all_files_exist()
            data_transformation.convert()
        except Exception as e:
            raise e