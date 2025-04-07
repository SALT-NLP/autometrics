import warnings
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Optional
from autometrics.metrics.MultiMetric import MultiMetric
from autometrics.metrics.Metric import Metric

class Dataset(BaseModel):
    """
    Dataset class for handling and manipulating datasets.
    """
    dataframe: pd.DataFrame
    target_columns: List[str]
    ignore_columns: List[str]
    metric_columns: List[str]
    name: str
    data_id_column: Optional[str] = None
    model_id_column: Optional[str] = None
    input_column: Optional[str] = None
    output_column: Optional[str] = None
    reference_columns: Optional[List[str]] = None
    metrics: List[Metric] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
    
    def get_dataframe(self) -> pd.DataFrame:
        return self.dataframe
    
    def get_target_columns(self) -> List[str]:
        return self.target_columns
    
    def get_ignore_columns(self) -> List[str]:
        return self.ignore_columns
    
    def get_metric_columns(self) -> List[str]:
        return self.metric_columns
    
    def get_name(self) -> str:
        return self.name
    
    def get_data_id_column(self) -> Optional[str]:
        return self.data_id_column
    
    def get_model_id_column(self) -> Optional[str]:
        return self.model_id_column
    
    def get_input_column(self) -> Optional[str]:
        return self.input_column
    
    def get_output_column(self) -> Optional[str]:
        return self.output_column
    
    def get_reference_columns(self) -> Optional[List[str]]:
        return self.reference_columns
    
    def get_metrics(self) -> List[Metric]:
        return self.metrics
    
    def set_dataframe(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def add_metric(self, metric: Metric, update_dataset: bool = True):
        if isinstance(metric, MultiMetric):
            self.metrics.append(metric)
            for submetric_name in metric.get_submetric_names():
                if submetric_name not in self.metric_columns:
                    self.metric_columns.append(submetric_name)

            if self.dataframe is not None and update_dataset and any(submetric_name not in self.dataframe.columns for submetric_name in metric.get_submetric_names()):
                metric.predict(self, update_dataset=update_dataset)
        else:
            self.metrics.append(metric)
            if metric.get_name() not in self.metric_columns:
                self.metric_columns.append(metric.get_name())
            if self.dataframe is not None and update_dataset and metric.get_name() not in self.dataframe.columns:
                metric.predict(self, update_dataset=update_dataset)

    def add_metrics(self, metrics: List[Metric], update_dataset: bool = True):
        for metric in metrics:
            self.add_metric(metric, update_dataset=update_dataset)
    
    def __str__(self) -> str:
        return (f"Dataset: {self.name}, Target Columns: {self.target_columns}, "
                f"Ignore Columns: {self.ignore_columns}, Metric Columns: {self.metric_columns}\n"
                f"{self.dataframe.head()}")
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def get_splits(self, split_column: Optional[str] = None, train_ratio: float = 0.5, val_ratio: float = 0.2, seed: Optional[int] = None, max_size: Optional[int] = None):
        df = self.get_dataframe()

        if not split_column:
            split_column = self.data_id_column
        if not split_column:
            warnings.warn("No split column specified. Splitting based on index which is not recommended. "
                          "This means that we could be testing on data that is partially represented in the training set due to rows with similar data but different indices.")
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

        train_dataset = Dataset(
            dataframe=train_df,
            target_columns=self.target_columns,
            ignore_columns=self.ignore_columns,
            metric_columns=self.metric_columns,
            name=self.name,
            data_id_column=self.data_id_column,
            model_id_column=self.model_id_column,
            input_column=self.input_column,
            output_column=self.output_column,
            reference_columns=self.reference_columns,
            metrics=[]
        )
        val_dataset = Dataset(
            dataframe=val_df,
            target_columns=self.target_columns,
            ignore_columns=self.ignore_columns,
            metric_columns=self.metric_columns,
            name=self.name,
            data_id_column=self.data_id_column,
            model_id_column=self.model_id_column,
            input_column=self.input_column,
            output_column=self.output_column,
            reference_columns=self.reference_columns,
            metrics=[]
        )
        test_dataset = Dataset(
            dataframe=test_df,
            target_columns=self.target_columns,
            ignore_columns=self.ignore_columns,
            metric_columns=self.metric_columns,
            name=self.name,
            data_id_column=self.data_id_column,
            model_id_column=self.model_id_column,
            input_column=self.input_column,
            output_column=self.output_column,
            reference_columns=self.reference_columns,
            metrics=[]
        )
        
        if max_size:
            train_dataset = train_dataset.get_subset(max_size)
            val_dataset = val_dataset.get_subset(max_size)
            test_dataset = test_dataset.get_subset(max_size)

        return train_dataset, val_dataset, test_dataset

    def get_kfold_splits(self, k: int = 5, split_column: Optional[str] = None, seed: Optional[int] = None, test_ratio: float = 0.3):
        df = self.get_dataframe()

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
                          "This means that we could be testing on data that is partially represented in the training set due to rows with similar data but different indices.")
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
            split_val_dataset = Dataset(
                dataframe=split_df,
                target_columns=self.target_columns,
                ignore_columns=self.ignore_columns,
                metric_columns=self.metric_columns,
                name=self.name,
                data_id_column=self.data_id_column,
                model_id_column=self.model_id_column,
                input_column=self.input_column,
                output_column=self.output_column,
                reference_columns=self.reference_columns,
                metrics=[]
            )
            split_train_dataset = Dataset(
                dataframe=non_split_df,
                target_columns=self.target_columns,
                ignore_columns=self.ignore_columns,
                metric_columns=self.metric_columns,
                name=self.name,
                data_id_column=self.data_id_column,
                model_id_column=self.model_id_column,
                input_column=self.input_column,
                output_column=self.output_column,
                reference_columns=self.reference_columns,
                metrics=[]
            )
            split_datasets.append((split_train_dataset, split_val_dataset))

        train_dataset = Dataset(
            dataframe=train_df,
            target_columns=self.target_columns,
            ignore_columns=self.ignore_columns,
            metric_columns=self.metric_columns,
            name=self.name,
            data_id_column=self.data_id_column,
            model_id_column=self.model_id_column,
            input_column=self.input_column,
            output_column=self.output_column,
            reference_columns=self.reference_columns,
            metrics=[]
        )
        test_dataset = Dataset(
            dataframe=test_df,
            target_columns=self.target_columns,
            ignore_columns=self.ignore_columns,
            metric_columns=self.metric_columns,
            name=self.name,
            data_id_column=self.data_id_column,
            model_id_column=self.model_id_column,
            input_column=self.input_column,
            output_column=self.output_column,
            reference_columns=self.reference_columns,
            metrics=[]
        )

        return split_datasets, train_dataset, test_dataset
    
    def calculate_metrics(self, update_dataset: bool = True, **kwargs):
        for metric in self.metrics:
            if metric.get_name() not in self.get_metric_columns():
                metric.predict(self, update_dataset=update_dataset, **kwargs)

            df = self.get_dataframe()

            for i, row in df.iterrows():
                if metric.get_name() not in row:
                    metric.calculate_row(row, self, update_dataset=update_dataset)

    def get_subset(self, size: int, seed: Optional[int] = None) -> 'Dataset':
        df = self.get_dataframe()
        if seed:
            np.random.seed(seed)
        subset_df = df.sample(n=size)
        return Dataset(
            dataframe=subset_df,
            target_columns=self.target_columns,
            ignore_columns=self.ignore_columns,
            metric_columns=self.metric_columns,
            name=self.name,
            data_id_column=self.data_id_column,
            model_id_column=self.model_id_column,
            input_column=self.input_column,
            output_column=self.output_column,
            reference_columns=self.reference_columns,
            metrics=[]
        )
    
    def copy(self) -> 'Dataset':
        return Dataset(
            dataframe=self.dataframe.copy(),
            target_columns=self.target_columns.copy(),
            ignore_columns=self.ignore_columns.copy(),
            metric_columns=self.metric_columns.copy(),
            name=self.name,
            data_id_column=self.data_id_column,
            model_id_column=self.model_id_column,
            input_column=self.input_column,
            output_column=self.output_column,
            reference_columns=self.reference_columns.copy(),
            metrics=[metric for metric in self.metrics]
        )