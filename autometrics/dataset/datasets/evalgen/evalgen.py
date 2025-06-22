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

    def get_splits(self, split_column: Optional[str] = None, train_ratio: float = 0.5, val_ratio: float = 0.2, seed: Optional[int] = None, max_size: Optional[int] = None, *, preserve_splits=True):
        """
        Get the train, validation, and test splits of the dataset.  We will use the predefined splits in the dataset which include "train" and "test".  We need to add validation.
        """

        if preserve_splits:
            # When preserving splits, always use the predefined split column regardless of input
            actual_split_column = "Metavar: split"

            # Get the train and test splits
            train_df = self.dataframe[self.dataframe[actual_split_column] == "train"].copy()
            test_df = self.dataframe[self.dataframe[actual_split_column] == "test"].copy()

            # Check if we have any data
            if len(train_df) == 0 and len(test_df) == 0:
                raise ValueError("No train or test data found in the dataset")
            
            # If max_size is provided, we will sample the dataframe to that size
            if max_size is not None:
                if len(train_df) > 0:
                    train_df = train_df.sample(n=min(max_size, len(train_df)), random_state=seed)
                if len(test_df) > 0:
                    test_df = test_df.sample(n=min(max_size, len(test_df)), random_state=seed)

            # Create test dataset
            if len(test_df) > 0:
                test_dataset = Dataset(
                    test_df, self.target_columns, self.ignore_columns, self.metric_columns, 
                    self.name, self.data_id_column, self.model_id_column, self.input_column, 
                    self.output_column, self.reference_columns, self.metrics, 
                    task_description=self.task_description
                )
            else:
                test_dataset = None

            # Split the training data into train and validation
            if len(train_df) > 0:
                # Calculate adjusted ratios for splitting the training data only
                total_train_val_ratio = train_ratio + val_ratio
                adjusted_train_ratio = train_ratio / total_train_val_ratio
                adjusted_val_ratio = val_ratio / total_train_val_ratio
                
                # Get the validation split from training data
                val_df = train_df.sample(frac=adjusted_val_ratio, random_state=seed)
                train_df = train_df.drop(val_df.index)

                train_dataset = Dataset(
                    train_df, self.target_columns, self.ignore_columns, self.metric_columns, 
                    self.name, self.data_id_column, self.model_id_column, self.input_column, 
                    self.output_column, self.reference_columns, self.metrics,
                    task_description=self.task_description
                )
                val_dataset = Dataset(
                    val_df, self.target_columns, self.ignore_columns, self.metric_columns, 
                    self.name, self.data_id_column, self.model_id_column, self.input_column, 
                    self.output_column, self.reference_columns, self.metrics,
                    task_description=self.task_description
                )
            else:
                # No training data - create empty datasets
                empty_df = self.dataframe.iloc[0:0].copy()
                train_dataset = Dataset(
                    empty_df, self.target_columns, self.ignore_columns, self.metric_columns, 
                    self.name, self.data_id_column, self.model_id_column, self.input_column, 
                    self.output_column, self.reference_columns, self.metrics,
                    task_description=self.task_description
                )
                val_dataset = Dataset(
                    empty_df, self.target_columns, self.ignore_columns, self.metric_columns, 
                    self.name, self.data_id_column, self.model_id_column, self.input_column, 
                    self.output_column, self.reference_columns, self.metrics,
                    task_description=self.task_description
                )

            return train_dataset, val_dataset, test_dataset
        else:
            return super().get_splits(split_column, train_ratio, val_ratio, seed, max_size=max_size)
    
    def get_kfold_splits(self, k: int = 5, split_column: Optional[str] = None, seed: Optional[int] = None, test_ratio: float = 0.3, max_size: Optional[int] = None, *, preserve_splits=True):
        """
        Get the k-fold splits of the dataset.  We will use the predefined splits in the dataset which include "train" and "test".  We need to add validation.
        """

        if preserve_splits:
            # When preserving splits, always use the predefined split column regardless of input
            actual_split_column = "Metavar: split"

            # Get the train and test splits
            train_df = self.dataframe[self.dataframe[actual_split_column] == "train"].copy()
            test_df = self.dataframe[self.dataframe[actual_split_column] == "test"].copy()

            # Apply max_size to test set FIRST to ensure consistency with regular splits
            if max_size is not None and len(test_df) > 0:
                test_df = test_df.sample(n=min(max_size, len(test_df)), random_state=seed)

            # Create test dataset (using the predefined test set)
            if len(test_df) > 0:
                test_dataset = Dataset(
                    test_df, self.target_columns, self.ignore_columns, self.metric_columns, 
                    self.name, self.data_id_column, self.model_id_column, self.input_column, 
                    self.output_column, self.reference_columns, self.metrics,
                    task_description=self.task_description
                )
            else:
                test_dataset = None

            # Create k-fold splits from the training data only (no separate test extraction)
            if len(train_df) > 0:
                # Use the parent class k-fold method on the training data
                train_dataset_temp = Dataset(
                    train_df, self.target_columns, self.ignore_columns, self.metric_columns, 
                    self.name, self.data_id_column, self.model_id_column, self.input_column, 
                    self.output_column, self.reference_columns, self.metrics,
                    task_description=self.task_description
                )
                
                splits, full_train_dataset, _ = train_dataset_temp.get_kfold_splits(
                    k=k, split_column=self.data_id_column, seed=seed, test_ratio=0.0, max_size=max_size
                )
            else:
                # No training data - create empty datasets
                empty_df = self.dataframe.iloc[0:0].copy()
                full_train_dataset = Dataset(
                    empty_df, self.target_columns, self.ignore_columns, self.metric_columns, 
                    self.name, self.data_id_column, self.model_id_column, self.input_column, 
                    self.output_column, self.reference_columns, self.metrics,
                    task_description=self.task_description
                )
                splits = []

            return splits, full_train_dataset, test_dataset
        
        else:
            return super().get_kfold_splits(k, split_column, seed, test_ratio, max_size)

class EvalGenProduct(EvalGen):
    def __init__(self, path='./autometrics/dataset/datasets/evalgen/product.csv'):
        super().__init__(path)
        self.name = "evalgen_product"
        self.task_description = """You are an expert copywriter. You need to write an e-commerce product description based on the product details and customer reviews. Your description should be SEO-optimized. It should use an active voice and include the product's features, benefits, unique selling points without overpromising, and a call to action for the buyer. Benefits describe how product features will work for the buyer, addressing exactly how the product will improve their lives. Clearly distinguish between features (e.g., lightweight, USB-chargeable) and benefits (e.g., convenience, nutritious drinks on-the-go). Don't mention weaknesses of the product or use generic or repetitive language. Don't make up review text or quotes. Don't include any links. Don't cite the reviews too heavily. Divide your description into readable chunks divided by relevant subheadings. Keep your description around 200 words, no more than 300, in Markdown format.\n\n"""

class EvalGenMedical(EvalGen):
    def __init__(self, path='./autometrics/dataset/datasets/evalgen/medical.csv'):
        super().__init__(path)
        self.name = "evalgen_medical"
        self.task_description = """You are extracting insights from some medical records. The records contain a medical note and a dialogue between a doctor and a patient. You need to extract values for the following: Chief complaint, History of present illness, Physical examination, Symptoms experienced by the patient, New medications prescribed or changed, including dosages (N/A if not provided), and Follow-up instructions (N/A if not provided). Your answer should not include any personal identifiable information (PII) such as name, age, gender, or ID. Use "the patient" instead of their name, for example. Return your answer as a bullet list, where each bullet is formatted like `chief complaint: xx.` If there is no value for the key, the value should be `N/A`. Keep your response around 150 words (you may have to summarize some extracted values to stay within the word limit)."""