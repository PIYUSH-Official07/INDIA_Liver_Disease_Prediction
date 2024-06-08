import os
import sys

import numpy as np
import pandas as pd
from liver_disease.entity.config_entity import INliverPredictorConfig
from liver_disease.entity.s3_estimator import INliverEstimator
from liver_disease.exception import INliverException
from liver_disease.logger import logging
from liver_disease.utils.main_utils import read_yaml_file
from pandas import DataFrame


class INDliverData:
    def __init__(self,
                Age,
                Gender,
                Total_Bilirubin,
                Direct_Bilirubin,
                Alkaline_Phosphotase,
                Alamine_Aminotransferase,
                Aspartate_Aminotransferase,
                Total_Protiens,
                Albumin,
                Albumin_and_Globulin_Ratio
                ):
        """
        INDliver Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.Age = Age
            self.Gender = Gender
            self.Total_Bilirubin = Total_Bilirubin
            self.Direct_Bilirubin = Direct_Bilirubin
            self.Alkaline_Phosphotase = Alkaline_Phosphotase
            self.Alamine_Aminotransferase = Alamine_Aminotransferase
            self.Aspartate_Aminotransferase = Aspartate_Aminotransferase
            self.Total_Protiens = Total_Protiens
            self.Albumin = Albumin
            self.Albumin_and_Globulin_Ratio = Albumin_and_Globulin_Ratio


        except Exception as e:
            raise INliverException(e, sys) from e

    def get_indliver_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from USvisaData class input
        """
        try:
            
            indliver_input_dict = self.get_indliver_data_as_dict()
            return DataFrame(indliver_input_dict)
        
        except Exception as e:
            raise INliverException(e, sys) from e
        
    def get_indliver_data_as_dict(self):
        """
        This function returns a dictionary from INDliverData class input 
        """
        logging.info("Entered get_indliver_data_as_dict method as INDliverData class")

        try:
            input_data = {
                "Age": [self.Age],
                "Gender": [self.Gender],
                "Total_Bilirubin": [self.Total_Bilirubin],
                "Direct_Bilirubin": [self.Direct_Bilirubin],
                "Alkaline_Phosphotase": [self.Alkaline_Phosphotase],
                "Alamine_Aminotransferase": [self.Alamine_Aminotransferase],
                "Aspartate_Aminotransferase": [self.Aspartate_Aminotransferase],
                "Total_Protiens": [self.Total_Protiens],
                "Albumin": [self.Albumin],
                "Albumin_and_Globulin_Ratio": [self.Albumin_and_Globulin_Ratio]
            }

            logging.info("Created indliver data dict")

            logging.info("Exited get_indliver_data_as_dict method as INDliveraData class")

            return input_data

        except Exception as e:
            raise INliverException(e, sys) from e

class INDliverClassifier:
    def __init__(self,prediction_pipeline_config: INliverPredictorConfig = INliverPredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            # self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise INliverException(e, sys)

    def predict(self, dataframe) -> str:
        """
        This is the method of INDliverClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of USvisaClassifier class")
            model = INliverEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result =  model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise INliverException(e, sys)