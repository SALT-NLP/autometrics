import warnings
import numpy as np
import pandas as pd
from typing import List, Optional
from pydantic import Field
from autometrics.dataset.Dataset import Dataset
from autometrics.metrics.MultiMetric import MultiMetric
from autometrics.metrics.Metric import Metric

class PairwiseDataset(Dataset):
    """
    Dataset class for handling and manipulating pairwise datasets.
    """
    # target_columns_1: List[str] = Field(default_factory=list)
    # target_columns_2: List[str] = Field(default_factory=list)
    # metric_columns_1: List[str] = Field(default_factory=list)
    # metric_columns_2: List[str] = Field(default_factory=list)
    # model_id_column_1: Optional[str] = None
    # model_id_column_2: Optional[str] = None
    # output_column_1: Optional[str] = None
    # output_column_2: Optional[str] = None
    # model1_dataset: Optional[Dataset] = None
    # model2_dataset: Optional[Dataset] = None

    def __init__(self, dataframe: pd.DataFrame, target_columns_1: List[str], target_columns_2: List[str], ignore_columns: List[str], metric_columns_1: List[str], metric_columns_2: List[str], name: str, data_id_column: Optional[str] = None, model_id_column_1: Optional[str] = None, model_id_column_2: Optional[str] = None, input_column: Optional[str] = None, output_column_1: Optional[str] = None, output_column_2: Optional[str] = None, reference_columns: Optional[List[str]] = None, metrics: List[Metric] = None):
        assert len(target_columns_1) == len(target_columns_2), "Target columns for both models must be the same length"
        assert len(metric_columns_1) == len(metric_columns_2), "Metric columns for both models must be the same length"

        # PairwiseDataset specific attributes
        self.target_columns_1 = target_columns_1
        self.target_columns_2 = target_columns_2
        self.metric_columns_1 = metric_columns_1
        self.metric_columns_2 = metric_columns_2
        self.model_id_column_1 = model_id_column_1
        self.model_id_column_2 = model_id_column_2
        self.output_column_1 = output_column_1
        self.output_column_2 = output_column_2

        # Step 1: Create model1_dataset and model2_dataset by splitting the dataframe

        model_1_columns = target_columns_1 + metric_columns_1 + [data_id_column, model_id_column_1, input_column, output_column_1] + reference_columns
        model_2_columns = target_columns_2 + metric_columns_2 + [data_id_column, model_id_column_2, input_column, output_column_2] + reference_columns

        model_1_columns = list(set(model_1_columns))
        model_2_columns = list(set(model_2_columns))

        model_1_df = dataframe[model_1_columns].copy()
        model_2_df = dataframe[model_2_columns].copy()

        self.model1_dataset = Dataset(
            dataframe=model_1_df,
            target_columns=target_columns_1,
            ignore_columns=ignore_columns,
            metric_columns=metric_columns_1,
            name=name,
            data_id_column=data_id_column,
            model_id_column=model_id_column_1,
            input_column=input_column,
            output_column=output_column_1,
            reference_columns=reference_columns,
            metrics=metrics
        )

        self.model2_dataset = Dataset(
            dataframe=model_2_df,
            target_columns=target_columns_2,
            ignore_columns=ignore_columns,
            metric_columns=metric_columns_2,
            name=name,
            data_id_column=data_id_column,
            model_id_column=model_id_column_2,
            input_column=input_column,
            output_column=output_column_2,
            reference_columns=reference_columns,
            metrics=metrics
        )

        # Step 2: Set the dataframe and other attributes for the PairwiseDataset to be a combination of the two datasets
        self.target_columns = [f"{col1}-{col2}" for col1, col2 in zip(target_columns_1, target_columns_2)]
        self.ignore_columns = ignore_columns
        self.metric_columns = [f"{col1}-{col2}" for col1, col2 in zip(metric_columns_1, metric_columns_2)]
        self.name = name
        self.data_id_column = data_id_column
        self.model_id_column = "model_id"  # Set a common model_id column for the pairwise dataset
        self.input_column = input_column
        self.output_column = "output"  # Set a common output column for the pairwise dataset
        self.reference_columns = reference_columns
        self.metrics = metrics if metrics is not None else []

        # Combine the dataframes for the pairwise dataset
        cols = [data_id_column, input_column] + reference_columns

        cols = list(set(cols)) # Ensure unique columns

        df = dataframe[cols].copy()
        df.columns = cols
        # add model_id, output, and target, and metric columns
        df["model_id"] = model_1_df[model_id_column_1].astype(str) + "-" + model_2_df[model_id_column_2].astype(str)
        df["output"] = "[\"" + model_1_df[output_column_1].astype(str) + "\",\"" + model_2_df[output_column_2].astype(str) + "\"]"
        for col1, col2 in zip(target_columns_1, target_columns_2):
            df[f"{col1}-{col2}"] = model_1_df[col1].astype(float) - model_2_df[col2].astype(float)
        for col1, col2 in zip(metric_columns_1, metric_columns_2):
            df[f"{col1}-{col2}"] = model_1_df[col1].astype(float) - model_2_df[col2].astype(float)

        self.dataframe = df
        self.original_dataframe = dataframe.copy()

    def set_all_fields(self, dataframe: pd.DataFrame, target_columns_1: List[str], target_columns_2: List[str], ignore_columns: List[str], metric_columns_1: List[str], metric_columns_2: List[str], name: str, data_id_column: Optional[str] = None, model_id_column_1: Optional[str] = None, model_id_column_2: Optional[str] = None, input_column: Optional[str] = None, output_column_1: Optional[str] = None, output_column_2: Optional[str] = None, reference_columns: Optional[List[str]] = None, metrics: List[Metric] = None):
        self.dataframe = dataframe
        self.original_dataframe = dataframe.copy()
        self.target_columns_1 = target_columns_1
        self.target_columns_2 = target_columns_2
        self.ignore_columns = ignore_columns
        self.metric_columns_1 = metric_columns_1
        self.metric_columns_2 = metric_columns_2
        self.name = name
        self.data_id_column = data_id_column
        self.model_id_column_1 = model_id_column_1
        self.model_id_column_2 = model_id_column_2
        self.input_column = input_column
        self.output_column_1 = output_column_1
        self.output_column_2 = output_column_2
        self.reference_columns = reference_columns if reference_columns is not None else []
        self.metrics = metrics if metrics is not None else []


    def set_dataframe(self, dataframe: pd.DataFrame):
        print("[WARNING] Setting dataframe directly. This is not recommended for pairwise datasets which have custom logic for handling data.")
        self.dataframe = dataframe

    def add_metric(self, metric: Metric, update_dataset: bool = True):
        self.model1_dataset.add_metric(metric, update_dataset=update_dataset)
        self.model2_dataset.add_metric(metric, update_dataset=update_dataset)
        model1_df = self.model1_dataset.get_dataframe()
        model2_df = self.model2_dataset.get_dataframe()

        if isinstance(metric, MultiMetric):
            self.metrics.append(metric)
            for submetric_name in metric.get_submetric_names():
                if submetric_name not in self.metric_columns:
                    self.metric_columns.append(submetric_name)

                if self.dataframe is not None and update_dataset and submetric_name not in self.dataframe.columns:
                    self.dataframe[submetric_name] = model1_df[submetric_name].astype(float) - model2_df[submetric_name].astype(float)

        else:
            self.metrics.append(metric)
            metric_name = metric.get_name()
            if metric_name not in self.metric_columns:
                self.metric_columns.append(metric_name)
            if self.dataframe is not None and update_dataset and metric_name not in self.dataframe.columns:
                self.dataframe[metric_name] = model1_df[metric_name].astype(float) - model2_df[metric_name].astype(float)
            
    
    def get_splits(self, split_column: Optional[str] = None, train_ratio: float = 0.5, val_ratio: float = 0.2, seed: Optional[int] = None, max_size: Optional[int] = None):
        df = self.original_dataframe

        if len(self.metrics) > 0:
            warnings.warn("Metrics have already been added to this dataset.  For PairwiseDataset, metrics must be added AFTER splitting the dataset."
                            "DISCARDING Metrics added to this dataset.")

        if not split_column:
            split_column = self.data_id_column
        if not split_column:
            warnings.warn("No split column specified. Splitting based on index which is not recommended. "
                          "This means that we could be testing on data that is partially represented in the training set.")
            items = np.arange(len(df))
        else:
            items = df[split_column].unique()

        train_size = int(train_ratio * len(items))
        val_size = int(val_ratio * len(items))
        test_size = len(items) - train_size - val_size

        if train_size + val_size + test_size < len(items):
            train_size += len(items) - (train_size + val_size + test_size)

        if seed:
            np.random.seed(seed)

        np.random.shuffle(items)

        train_items = items[:train_size]
        val_items = items[train_size:train_size + val_size]
        test_items = items[train_size + val_size:]

        if split_column:
            train_df = df[df[split_column].isin(train_items)]
            val_df = df[df[split_column].isin(val_items)]
            test_df = df[df[split_column].isin(test_items)]
        else:
            train_df = df.iloc[train_items]
            val_df = df.iloc[val_items]
            test_df = df.iloc[test_items]

        train_dataset = PairwiseDataset(
            dataframe=train_df,
            target_columns_1=self.target_columns_1,
            target_columns_2=self.target_columns_2,
            ignore_columns=self.ignore_columns,
            metric_columns_1=self.metric_columns_1,
            metric_columns_2=self.metric_columns_2,
            name=self.name,
            data_id_column=self.data_id_column,
            model_id_column_1=self.model_id_column_1,
            model_id_column_2=self.model_id_column_2,
            input_column=self.input_column,
            output_column_1=self.output_column_1,
            output_column_2=self.output_column_2,
            reference_columns=self.reference_columns,
            metrics=[]
        )
        val_dataset = PairwiseDataset(
            dataframe=val_df,
            target_columns_1=self.target_columns_1,
            target_columns_2=self.target_columns_2,
            ignore_columns=self.ignore_columns,
            metric_columns_1=self.metric_columns_1,
            metric_columns_2=self.metric_columns_2,
            name=self.name,
            data_id_column=self.data_id_column,
            model_id_column_1=self.model_id_column_1,
            model_id_column_2=self.model_id_column_2,
            input_column=self.input_column,
            output_column_1=self.output_column_1,
            output_column_2=self.output_column_2,
            reference_columns=self.reference_columns,
            metrics=[]
        )
        test_dataset = PairwiseDataset(
            dataframe=test_df,
            target_columns_1=self.target_columns_1,
            target_columns_2=self.target_columns_2,
            ignore_columns=self.ignore_columns,
            metric_columns_1=self.metric_columns_1,
            metric_columns_2=self.metric_columns_2,
            name=self.name,
            data_id_column=self.data_id_column,
            model_id_column_1=self.model_id_column_1,
            model_id_column_2=self.model_id_column_2,
            input_column=self.input_column,
            output_column_1=self.output_column_1,
            output_column_2=self.output_column_2,
            reference_columns=self.reference_columns,
            metrics=[]
        )
        
        if max_size:
            train_dataset = train_dataset.get_subset(max_size)
            val_dataset = val_dataset.get_subset(max_size)
            test_dataset = test_dataset.get_subset(max_size)

        return train_dataset, val_dataset, test_dataset

    def get_kfold_splits(self, k: int = 5, split_column: Optional[str] = None, seed: Optional[int] = None, test_ratio: float = 0.3):
        df = self.original_dataframe

        if len(self.metrics) > 0:
            warnings.warn("Metrics have already been added to this dataset.  For PairwiseDataset, metrics must be added AFTER splitting the dataset."
                            "DISCARDING Metrics added to this dataset.")

        if test_ratio and test_ratio > 0:
            if seed:
                np.random.seed(seed)

            if not split_column:
                split_column = self.data_id_column

            if not split_column:
                items = np.arange(len(df))
                test_size = int(test_ratio * len(items))
                test_items = np.random.choice(np.arange(len(df)), test_size, replace=False)
                test_df = df[df.index.isin(test_items)]
                df = df[~df.index.isin(test_items)]
                train_df = df.copy()
            else:
                items = df[split_column].unique()
                test_size = int(test_ratio * len(items))
                test_items = np.random.choice(df[split_column].unique(), test_size, replace=False)
                test_df = df[df[split_column].isin(test_items)]
                df = df[~df[split_column].isin(test_items)]
                train_df = df.copy()
        else:
            test_df = None

        if not split_column:
            split_column = self.data_id_column
        if not split_column:
            warnings.warn("No split column specified. Splitting based on index which is not recommended. "
                          "This means that we could be testing on data that is partially represented in the training set.")
            items = np.arange(len(df))
        else:
            items = df[split_column].unique()

        if seed:
            np.random.seed(seed)

        np.random.shuffle(items)

        splits = np.array_split(items, k)

        split_datasets = []
        for i in range(k):
            split_items = splits[i]
            non_split_items = np.concatenate([splits[j] for j in range(k) if j != i])
            if split_column:
                split_df = df[df[split_column].isin(split_items)]
                non_split_df = df[df[split_column].isin(non_split_items)]
            else:
                split_df = df.iloc[split_items].copy()
                non_split_df = df.iloc[non_split_items].copy()
            split_val_dataset = PairwiseDataset(
                dataframe=split_df,
                target_columns_1=self.target_columns_1,
                target_columns_2=self.target_columns_2,
                ignore_columns=self.ignore_columns,
                metric_columns_1=self.metric_columns_1,
                metric_columns_2=self.metric_columns_2,
                name=self.name,
                data_id_column=self.data_id_column,
                model_id_column_1=self.model_id_column_1,
                model_id_column_2=self.model_id_column_2,
                input_column=self.input_column,
                output_column_1=self.output_column_1,
                output_column_2=self.output_column_2,
                reference_columns=self.reference_columns,
                metrics=[]
            )
            split_train_dataset = PairwiseDataset(
                dataframe=non_split_df,
                target_columns_1=self.target_columns_1,
                target_columns_2=self.target_columns_2,
                ignore_columns=self.ignore_columns,
                metric_columns_1=self.metric_columns_1,
                metric_columns_2=self.metric_columns_2,
                name=self.name,
                data_id_column=self.data_id_column,
                model_id_column_1=self.model_id_column_1,
                model_id_column_2=self.model_id_column_2,
                input_column=self.input_column,
                output_column_1=self.output_column_1,
                output_column_2=self.output_column_2,
                reference_columns=self.reference_columns,
                metrics=[]
            )
            split_datasets.append((split_train_dataset, split_val_dataset))

        train_dataset = PairwiseDataset(
            dataframe=train_df,
            target_columns_1=self.target_columns_1,
            target_columns_2=self.target_columns_2,
            ignore_columns=self.ignore_columns,
            metric_columns_1=self.metric_columns_1,
            metric_columns_2=self.metric_columns_2,
            name=self.name,
            data_id_column=self.data_id_column,
            model_id_column_1=self.model_id_column_1,
            model_id_column_2=self.model_id_column_2,
            input_column=self.input_column,
            output_column_1=self.output_column_1,
            output_column_2=self.output_column_2,
            reference_columns=self.reference_columns,
            metrics=[]
        )
        test_dataset = PairwiseDataset( 
            dataframe=test_df,
            target_columns_1=self.target_columns_1,
            target_columns_2=self.target_columns_2,
            ignore_columns=self.ignore_columns,
            metric_columns_1=self.metric_columns_1,
            metric_columns_2=self.metric_columns_2,
            name=self.name,
            data_id_column=self.data_id_column,
            model_id_column_1=self.model_id_column_1,
            model_id_column_2=self.model_id_column_2,
            input_column=self.input_column,
            output_column_1=self.output_column_1,
            output_column_2=self.output_column_2,
            reference_columns=self.reference_columns,
            metrics=[]
        )

        return split_datasets, train_dataset, test_dataset
    
    def calculate_metrics(self, update_dataset: bool = True, **kwargs):
        for metric in self.metrics:
            if metric.get_name() not in self.get_metric_columns():
                self.add_metric(metric, update_dataset=update_dataset)

            df = self.get_dataframe()

            for i, row in df.iterrows():
                if metric.get_name() not in row:
                    res1 = metric.calculate_row(row, self.model1_dataset, update_dataset=update_dataset)
                    res2 = metric.calculate_row(row, self.model2_dataset, update_dataset=update_dataset)

                    if update_dataset:
                        df.at[i, metric.get_name()] = res1 - res2

        if update_dataset:
            self.set_dataframe(df)

    def get_subset(self, size: int, seed: Optional[int] = None) -> 'PairwiseDataset':
        df = self.get_dataframe() if len(self.metrics) > 0 else self.original_dataframe
        if seed:
            np.random.seed(seed)
        subset_df = df.sample(n=min(size, len(df)), random_state=seed)

        if len(self.metrics) == 0:
            return PairwiseDataset(
                dataframe=subset_df,
                target_columns_1=self.target_columns_1,
                target_columns_2=self.target_columns_2,
                ignore_columns=self.ignore_columns,
                metric_columns_1=self.metric_columns_1,
                metric_columns_2=self.metric_columns_2,
                name=self.name,
                data_id_column=self.data_id_column,
                model_id_column_1=self.model_id_column_1,
                model_id_column_2=self.model_id_column_2,
                input_column=self.input_column,
                output_column_1=self.output_column_1,
                output_column_2=self.output_column_2,
                reference_columns=self.reference_columns,
                metrics=[]
            )

        indices = subset_df.index
        model1_df = self.model1_dataset.get_dataframe().loc[indices].copy()
        model2_df = self.model2_dataset.get_dataframe().loc[indices].copy()

        model1_dataset = self.model1_dataset.copy()
        model1_dataset.set_dataframe(model1_df)

        model2_dataset = self.model2_dataset.copy()
        model2_dataset.set_dataframe(model2_df)

        output_dataset = PairwiseDataset(
            dataframe=subset_df,
            target_columns_1=self.target_columns_1,
            target_columns_2=self.target_columns_2,
            ignore_columns=self.ignore_columns,
            metric_columns_1=self.metric_columns_1,
            metric_columns_2=self.metric_columns_2,
            name=self.name,
            data_id_column=self.data_id_column,
            model_id_column_1=self.model_id_column_1,
            model_id_column_2=self.model_id_column_2,
            input_column=self.input_column,
            output_column_1=self.output_column_1,
            output_column_2=self.output_column_2,
            reference_columns=self.reference_columns,
            metrics=[],
        )

        output_dataset.set_all_fields(
            dataframe=subset_df,
            target_columns_1=self.target_columns_1,
            target_columns_2=self.target_columns_2,
            ignore_columns=self.ignore_columns,
            metric_columns_1=self.metric_columns_1,
            metric_columns_2=self.metric_columns_2,
            name=self.name,
            data_id_column=self.data_id_column,
            model_id_column_1=self.model_id_column_1,
            model_id_column_2=self.model_id_column_2,
            input_column=self.input_column,
            output_column_1=self.output_column_1,
            output_column_2=self.output_column_2,
            reference_columns=self.reference_columns,
            metrics=[],
            model1_dataset=model1_dataset,
            model2_dataset=model2_dataset
        )
    
    def copy(self):
        new_dataset = PairwiseDataset(
            dataframe=self.dataframe.copy(),
            target_columns_1=self.target_columns_1,
            target_columns_2=self.target_columns_2,
            ignore_columns=self.ignore_columns,
            metric_columns_1=self.metric_columns_1,
            metric_columns_2=self.metric_columns_2,
            name=self.name,
            data_id_column=self.data_id_column,
            model_id_column_1=self.model_id_column_1,
            model_id_column_2=self.model_id_column_2,
            input_column=self.input_column,
            output_column_1=self.output_column_1,
            output_column_2=self.output_column_2,
            reference_columns=self.reference_columns,
            metrics=[]
        )
        new_dataset.set_all_fields(
            dataframe=self.dataframe.copy(),
            target_columns_1=self.target_columns_1,
            target_columns_2=self.target_columns_2,
            ignore_columns=self.ignore_columns,
            metric_columns_1=self.metric_columns_1,
            metric_columns_2=self.metric_columns_2,
            name=self.name,
            data_id_column=self.data_id_column,
            model_id_column_1=self.model_id_column_1,
            model_id_column_2=self.model_id_column_2,
            input_column=self.input_column,
            output_column_1=self.output_column_1,
            output_column_2=self.output_column_2,
            reference_columns=self.reference_columns,
            metrics=self.metrics,
            model1_dataset=self.model1_dataset.copy(),
            model2_dataset=self.model2_dataset.copy()
        )
        return new_dataset