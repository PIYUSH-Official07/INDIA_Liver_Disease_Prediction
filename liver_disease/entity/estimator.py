import sys
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from liver_disease.exception import INliverException
from liver_disease.logger import logging
import pandas as pd

class TargetValueMapping:
    def __init__(self):
        self.value_1:int = 0
        self.value_2:int = 1

    def _asdict(self):
        return self.__dict__

    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))

class INDliverModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def handle_missing_values(self, dataframe: DataFrame) -> DataFrame:
        # Replace missing values with the most frequent category for each column
        for column in dataframe.select_dtypes(include=['object']).columns:
            most_frequent_category = dataframe[column].mode()[0]  # Get most frequent category
            dataframe[column].fillna(most_frequent_category, inplace=True)
        return dataframe

    def align_categories(self, dataframe: DataFrame, column: str, valid_categories: list) -> DataFrame:
        # Map unknown categories to the most frequent or default known category
        most_frequent_category = valid_categories[0]
        dataframe[column] = dataframe[column].apply(
            lambda x: x if x in valid_categories else most_frequent_category)
        return dataframe

    def predict(self, dataframe: DataFrame) -> DataFrame:
        logging.info("Entered predict method of INDliverModel class")

        try:
            # Handle missing values
            dataframe = self.handle_missing_values(dataframe)

            # Align categories to those seen during training
            valid_categories = ['Male', 'Female']  # Replace with actual categories from training data
            dataframe = self.align_categories(dataframe, 'Gender', valid_categories)

            logging.info("Transforming features")
            transformed_feature = self.preprocessing_object.transform(dataframe)

            logging.info("Using the trained model to get predictions")
            return self.trained_model_object.predict(transformed_feature)

        except Exception as e:
            raise INliverException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"
