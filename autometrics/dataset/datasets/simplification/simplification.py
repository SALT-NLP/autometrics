from autometrics.dataset.Dataset import Dataset
import pandas as pd

from autometrics.metrics.dummy import DummyMetric

# https://github.com/Yao-Dou/LENS

class SimpDA(Dataset):
    def __init__(self, path='./autometrics/dataset/datasets/simplification/simpda.csv'):
        df = pd.read_csv(path)


        df.drop(columns=['WorkerId'], inplace=True)

        target_columns = ['Answer.adequacy','Answer.fluency','Answer.simplicity']
        ignore_columns = ["Input.id","Input.original","Input.simplified","Input.system","ref1"]
        metric_columns = [col for col in df.columns if col not in target_columns and col not in ignore_columns]

        name = "SimpDA"

        data_id_column = "Input.id"
        model_id_column = "Input.system"
        input_column = "Input.original"
        output_column = "Input.simplified"
        reference_columns = ["ref1"]

        metrics = [DummyMetric(col) for col in metric_columns]

        super().__init__(df, target_columns, ignore_columns, metric_columns, name, data_id_column, model_id_column, input_column, output_column, reference_columns, metrics)

class SimpEval(Dataset):
    def __init__(self, path='./autometrics/dataset/datasets/simplification/simpeval.csv'):
        df = pd.read_csv(path)

        df['score'] = df[['rating_1','rating_2','rating_3']].mean(axis=1)

        df.drop(columns=['sentence_type','rating_1','rating_2','rating_3','rating_1_zscore','rating_2_zscore','rating_3_zscore'], inplace=True)

        target_columns = ['score']
        ignore_columns = ['original_id','original','generation','system','ref1']
        metric_columns = [col for col in df.columns if col not in target_columns and col not in ignore_columns]

        name = "SimpEval"

        data_id_column = "original_id"
        model_id_column = "system"
        input_column = "original"
        output_column = "generation"
        reference_columns = ["ref1"]

        metrics = [DummyMetric(col) for col in metric_columns]

        super().__init__(df, target_columns, ignore_columns, metric_columns, name, data_id_column, model_id_column, input_column, output_column, reference_columns, metrics)