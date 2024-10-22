import warnings
import numpy as np

class Dataset:
    """
    Dataset class for handling and manipulating datasets.
    Attributes:
        dataframe (pd.DataFrame): The dataframe containing the dataset.
        target_columns (list): List of columns that are target variables.
        ignore_columns (list): List of columns to be ignored in analysis.
        metric_columns (list): List of columns that are metrics.
        name (str): Name of the dataset.
    Methods:
        get_dataframe():
            Returns the dataframe of the dataset.
        get_target_columns():
            Returns the list of target columns.
        get_ignore_columns():
            Returns the list of columns to be ignored.
        get_metric_columns():
            Returns the list of metric columns.
        get_name():
            Returns the name of the dataset.
        __str__():
            Returns a string representation of the dataset, including its name, target columns, ignore columns, metric columns, and the head of the dataframe.
        __repr__():
            Returns a string representation of the dataset.
    """
    def __init__(self, dataframe, target_columns, ignore_columns, metric_columns, name, data_id_column=None, model_id_column=None, input_column=None, output_column=None, reference_columns=None, metrics=None):
        self.dataframe = dataframe
        self.target_columns = target_columns
        self.ignore_columns = ignore_columns
        self.metric_columns = metric_columns
        self.name = name
        self.data_id_column = data_id_column
        self.model_id_column = model_id_column
        self.input_column = input_column
        self.output_column = output_column
        self.reference_columns = reference_columns
        self.metrics = metrics

    def get_dataframe(self):
        return self.dataframe
    
    def get_target_columns(self):
        return self.target_columns
    
    def get_ignore_columns(self):
        return self.ignore_columns
    
    def get_metric_columns(self):
        return self.metric_columns
    
    def get_name(self):
        return self.name
    
    def get_data_id_column(self):
        return self.data_id_column
    
    def get_model_id_column(self):
        return self.model_id_column
    
    def get_input_column(self):
        return self.input_column
    
    def get_output_column(self):
        return self.output_column
    
    def get_reference_columns(self):
        return self.reference_columns
    
    def get_metrics(self):
        return self.metrics
    
    def set_dataframe(self, dataframe):
        self.dataframe = dataframe

    def add_metric(self, metric):
        self.metrics.append(metric)
        self.metric_columns.append(metric.get_name())
    
    def __str__(self):
        return f"Dataset: {self.name}, Target Columns: {self.target_columns}, Ignore Columns: {self.ignore_columns}, Metric Columns: {self.metric_columns}\n{self.dataframe.head()}"
    
    def __repr__(self):
        return self.__str__()
    
    def get_splits(self, split_column=None, train_ratio=0.5, val_ratio=0.2, seed=None):
        df = self.get_dataframe()

        if not split_column:
            split_column = self.data_id_column
        if not split_column:
            # Warning: We are going to split based on index which is not recommended.  This means that we could be testing on data that is partially represented in the training set.
            warnings.warn("No split column specified. Splitting based on index which is not recommended.  This means that we could be testing on data that is partially represented in the training set.")

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
        val_items = items[train_size:train_size+val_size]
        test_items = items[train_size+val_size:]

        if split_column:
            train_df = df[df[split_column].isin(train_items)]
            val_df = df[df[split_column].isin(val_items)]
            test_df = df[df[split_column].isin(test_items)]
        else:
            train_df = df.iloc[train_items]
            val_df = df.iloc[val_items]
            test_df = df.iloc[test_items]

        train_dataset = Dataset(train_df, self.target_columns, self.ignore_columns, self.metric_columns, self.name, self.data_id_column, self.model_id_column, self.input_column, self.output_column, self.reference_columns, self.metrics)
        val_dataset = Dataset(val_df, self.target_columns, self.ignore_columns, self.metric_columns, self.name, self.data_id_column, self.model_id_column, self.input_column, self.output_column, self.reference_columns, self.metrics)
        test_dataset = Dataset(test_df, self.target_columns, self.ignore_columns, self.metric_columns, self.name, self.data_id_column, self.model_id_column, self.input_column, self.output_column, self.reference_columns, self.metrics)
        
        return train_dataset, val_dataset, test_dataset

    def get_kfold_splits(self, k=5, split_column=None, seed=None, test_ratio=0.3):
        df = self.get_dataframe()

        if test_ratio and test_ratio > 0:
            # Sample test_ratio of the data for testing
            
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
            # Warning: We are going to split based on index which is not recommended.  This means that we could be testing on data that is partially represented in the training set.
            warnings.warn("No split column specified. Splitting based on index which is not recommended.  This means that we could be testing on data that is partially represented in the training set.")

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
            split_val_dataset = Dataset(split_df, self.target_columns, self.ignore_columns, self.metric_columns, self.name, self.data_id_column, self.model_id_column, self.input_column, self.output_column, self.reference_columns, self.metrics)
            split_train_dataset = Dataset(non_split_df, self.target_columns, self.ignore_columns, self.metric_columns, self.name, self.data_id_column, self.model_id_column, self.input_column, self.output_column, self.reference_columns, self.metrics)
            split_datasets.append((split_train_dataset, split_val_dataset))

        train_dataset = Dataset(train_df, self.target_columns, self.ignore_columns, self.metric_columns, self.name, self.data_id_column, self.model_id_column, self.input_column, self.output_column, self.reference_columns, self.metrics)
        test_dataset = Dataset(test_df, self.target_columns, self.ignore_columns, self.metric_columns, self.name, self.data_id_column, self.model_id_column, self.input_column, self.output_column, self.reference_columns, self.metrics)

        return split_datasets, train_dataset, test_dataset