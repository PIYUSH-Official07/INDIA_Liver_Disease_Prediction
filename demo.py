# from us_visa.logger import logging
# import sys
# from us_visa.exception import USvisaException

# #logging.info("Welcome to our Custom Log")
# try:
#     a= 1 / "10" 
# except Exception as e:
#     logging.info(e)
#     raise USvisaException(e,sys)

# from us_visa.constants import *
# print(DATABASE_NAME)

from liver_disease.pipeline.training_pipeline import TrainPipeline

pipeline = TrainPipeline()
pipeline.run_pipeline()