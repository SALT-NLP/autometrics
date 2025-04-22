from autometrics.dataset.Dataset import Dataset
import pandas as pd
from typing import Optional
from sklearn.model_selection import KFold

from autometrics.metrics.dummy import DummyMetric

class EvalGen(Dataset):
    def __init__(self, path='./autometrics/dataset/datasets/evalgen/product.csv'): # Path to the dataset: './autometrics/dataset/datasets/evalgen/medical.csv'
        #LLM,Prompt,Response,Response Batch Id,Var: document,Metavar: id,Metavar: split,Metavar: __pt,Metavar: LLM_0,grade,grading_feedback
        df = pd.read_csv(path)

        # Change the grade column to be 1 if it is "pass" (true) and 0 if it is "fail" (false)
        df['grade'] = df['grade'].apply(lambda x: 1 if x else 0)

        df.drop(columns=['Response Batch Id', 'Var: document', 'Metavar: __pt', 'Metavar: LLM_0'], inplace=True)

        target_columns = ['grade']
        ignore_columns = ["grading_feedback", "Metavar: split", "Metavar: id", "LLM", "Prompt", "Response"]
        metric_columns = []

        name = "evalgen"

        data_id_column = "Metavar: id"
        model_id_column = "LLM"
        input_column = "Prompt"
        output_column = "Response"
        reference_columns = []

        metrics = [DummyMetric(col) for col in metric_columns]

        super().__init__(df, target_columns, ignore_columns, metric_columns, name, data_id_column, model_id_column, input_column, output_column, reference_columns, metrics)

    def get_splits(self, split_column: Optional[str] = None, train_ratio: float = 0.5, val_ratio: float = 0.2, seed: Optional[int] = None, max_size: Optional[int] = None, preserve_splits=True):
        """
        Get the train, validation, and test splits of the dataset.  We will use the predefined splits in the dataset which include "train" and "test".  We need to add validation.
        """

        if preserve_splits:
            if not split_column:
                split_column = "Metavar: split"

            # Get the train and test splits
            train_df = self.dataframe[self.dataframe[split_column] == "train"]
            test_df = self.dataframe[self.dataframe[split_column] == "test"]

            # If max_size is provided, we will sample the dataframe to that size
            if max_size is not None:
                train_df = train_df.sample(n=min(max_size, len(train_df)), random_state=seed)
                test_df = test_df.sample(n=min(max_size, len(test_df)), random_state=seed)

            # Update the validation ratio since we are sampling directly from the trainset which is already a fraction of the original dataset
            true_train_ratio = len(train_df) / (len(train_df) + len(test_df))
            val_ratio = val_ratio * true_train_ratio

            # Get the validation split
            val_df = train_df.sample(frac=val_ratio, random_state=seed)
            train_df = train_df.drop(val_df.index)

            train_dataset = Dataset(train_df, self.target_columns, self.ignore_columns, self.metric_columns, self.name, self.data_id_column, self.model_id_column, self.input_column, self.output_column, self.reference_columns, self.metrics)
            val_dataset = Dataset(val_df, self.target_columns, self.ignore_columns, self.metric_columns, self.name, self.data_id_column, self.model_id_column, self.input_column, self.output_column, self.reference_columns, self.metrics)
            test_dataset = Dataset(test_df, self.target_columns, self.ignore_columns, self.metric_columns, self.name, self.data_id_column, self.model_id_column, self.input_column, self.output_column, self.reference_columns, self.metrics)

            return train_dataset, val_dataset, test_dataset
        else:
            return super().get_splits(split_column, train_ratio, val_ratio, seed, max_size=max_size)
    
    def get_kfold_splits(self, k: int = 5, split_column: Optional[str] = None, seed: Optional[int] = None, preserve_splits=True):
        """
        Get the k-fold splits of the dataset.  We will use the predefined splits in the dataset which include "train" and "test".  We need to add validation.
        """

        if preserve_splits:
            if not split_column:
                split_column = "Metavar: split"

            # Get the train and test splits
            train_df = self.dataframe[self.dataframe[split_column] == "train"]
            test_df = self.dataframe[self.dataframe[split_column] == "test"]

            # Get the k-fold splits
            kfold = KFold(n_splits=k, shuffle=True, random_state=seed)
            splits = list(kfold.split(train_df))

            # Create the datasets
            datasets = []
            for train_index, val_index in splits:
                train_split = train_df.iloc[train_index]
                val_split = train_df.iloc[val_index]

                train_dataset = Dataset(train_split, self.target_columns, self.ignore_columns, self.metric_columns, self.name, self.data_id_column, self.model_id_column, self.input_column, self.output_column, self.reference_columns, self.metrics)
                val_dataset = Dataset(val_split, self.target_columns, self.ignore_columns, self.metric_columns, self.name, self.data_id_column, self.model_id_column, self.input_column, self.output_column, self.reference_columns, self.metrics)

                datasets.append((train_dataset, val_dataset))

            full_train_dataset = Dataset(train_df, self.target_columns, self.ignore_columns, self.metric_columns, self.name, self.data_id_column, self.model_id_column, self.input_column, self.output_column, self.reference_columns, self.metrics)
            test_dataset = Dataset(test_df, self.target_columns, self.ignore_columns, self.metric_columns, self.name, self.data_id_column, self.model_id_column, self.input_column, self.output_column, self.reference_columns, self.metrics)

            return datasets, full_train_dataset, test_dataset
        
        else:
            return super().get_kfold_splits(k, split_column, seed)
    
class EvalGenProduct(EvalGen):
    def __init__(self, path='./autometrics/dataset/datasets/evalgen/product.csv'):
        super().__init__(path)
        self.name = "evalgen_product"

class EvalGenMedical(EvalGen):
    def __init__(self, path='./autometrics/dataset/datasets/evalgen/medical.csv'):
        super().__init__(path)
        self.name = "evalgen_medical"