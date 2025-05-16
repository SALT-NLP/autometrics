from autometrics.dataset.Dataset import Dataset
import pandas as pd

from autometrics.metrics.dummy import DummyMetric

# https://github.com/Yao-Dou/LENS

class SimpDA(Dataset):
    def __init__(self, path='./autometrics/dataset/datasets/simplification/simpda.csv'):
        df = pd.read_csv(path)

        # Split the reference columns into separate columns
        references = df['references'].apply(eval)

        # identify the longest reference list
        max_len = max(references.apply(len))

        for i in range(max_len):
            df[f'ref{i+1}'] = references.apply(lambda x: x[i] if len(x) > i else None)

        df.drop(columns=['references'], inplace=True)

        target_columns = ['fluency','meaning','simplicity']
        ignore_columns = ["id","original","simple","system"]
        ignore_columns.extend([f'ref{i+1}' for i in range(max_len)])
        metric_columns = [col for col in df.columns if col not in target_columns and col not in ignore_columns]

        name = "SimpDA"

        data_id_column = "id"
        model_id_column = "system"
        input_column = "original"
        output_column = "simple"
        reference_columns = [f'ref{i+1}' for i in range(max_len)]

        metrics = [DummyMetric(col) for col in metric_columns]

        task_description = """Given a complicated original sentence, simplify it in a way such that a broader audience could easily understand it."""

        super().__init__(df, target_columns, ignore_columns, metric_columns, name, data_id_column, model_id_column, input_column, output_column, reference_columns, metrics, task_description=task_description)

class SimpEval(Dataset):
    def __init__(self, path='./autometrics/dataset/datasets/simplification/simpeval.csv'):
        df = pd.read_csv(path)

        # Split the reference columns into separate columns
        references = df['references'].apply(eval)

        # identify the longest reference list
        max_len = max(references.apply(len))

        for i in range(max_len):
            df[f'ref{i+1}'] = references.apply(lambda x: x[i] if len(x) > i else None)

        df.drop(columns=['references'], inplace=True)

        target_columns = ['score']
        ignore_columns = ["id","original","simple","system"]
        ignore_columns.extend([f'ref{i+1}' for i in range(max_len)])
        metric_columns = [col for col in df.columns if col not in target_columns and col not in ignore_columns]

        name = "SimpEval"

        data_id_column = "id"
        model_id_column = "system"
        input_column = "original"
        output_column = "simple"
        reference_columns = [f'ref{i+1}' for i in range(max_len)]

        metrics = [DummyMetric(col) for col in metric_columns]

        task_description = """Given a complicated original sentence, simplify it in a way such that a broader audience could easily understand it."""

        super().__init__(df, target_columns, ignore_columns, metric_columns, name, data_id_column, model_id_column, input_column, output_column, reference_columns, metrics, task_description=task_description)