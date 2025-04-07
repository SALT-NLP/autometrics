import numpy as np
import pandas as pd
from typing import List, Optional
from pydantic import Field
from autometrics.dataset.PairwiseDataset import PairwiseDataset
from autometrics.metrics.Metric import Metric

class ChosenRejectedDataset(PairwiseDataset):
    """
    Dataset class for handling and manipulating datasets for chosen/rejected pairwise comparisons.
    """

    def __init__(dataframe: pd.DataFrame, chosen_rejected_columns: List[str], ignore_columns: List[str], metric_columns_1: List[str], metric_columns_2: List[str], name: str, data_id_column: Optional[str] = None, model_id_column_1: Optional[str] = None, model_id_column_2: Optional[str] = None, input_column: Optional[str] = None, output_column_1: Optional[str] = None, output_column_2: Optional[str] = None, reference_columns: Optional[List[str]] = None, metrics: List[Metric] = None):
        # The chosen_rejected columns are the target columns.  The expected format is either strings of the model names for which was chosen, or either 0 or 1 to indicate the chosen model.
        target_columns_1 = []
        target_columns_2 = []

        for column in chosen_rejected_columns:
            target_columns_1.append(column + "_1")
            target_columns_2.append(column + "_2")
            col_values_1, col_values_2 = [], []
            # Get the type of the column
            if dataframe[column].dtype == np.int64 or dataframe[column].dtype == np.float64:
                # If the column is numeric, then we assume it is a 0 or 1 column
                # We will build a series which is 1 if the value is 1, and 0 if the value is 0
                col_values_1 = dataframe[column].apply(lambda x: 1 if x == 0 else 0)
                col_values_2 = dataframe[column].apply(lambda x: 0 if x == 0 else 1)
            else:
                # If the column is not numeric, then we assume it is a string column
                # We will build a series which is 1 if the value is the model name, and 0 if it is not
                model_names_1 = dataframe[model_id_column_1].to_list()
                model_names_2 = dataframe[model_id_column_2].to_list()
                # We need to check if the model name matches for that exact row (not just that it is in the list)
                # To do this we will actually loop through the dataframe and check if the model name matches
                for i in range(len(dataframe)):
                    if dataframe[column][i] == model_names_1[i]:
                        col_values_1.append(1)
                    else:
                        col_values_1.append(0)
                    if dataframe[column][i] == model_names_2[i]:
                        col_values_2.append(1)
                    else:
                        col_values_2.append(0)
                # We need to convert the lists to series
                col_values_1 = pd.Series(col_values_1)
                col_values_2 = pd.Series(col_values_2)


            # Add the columns to the dataframe
            dataframe[column + "_1"] = col_values_1
            dataframe[column + "_2"] = col_values_2

        # Call the parent constructor
        super.__init__(dataframe, target_columns_1, target_columns_2, ignore_columns, metric_columns_1, metric_columns_2, name, data_id_column, model_id_column_1, model_id_column_2, input_column, output_column_1, output_column_2, reference_columns, metrics)
        

        