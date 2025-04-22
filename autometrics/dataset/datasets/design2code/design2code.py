from autometrics.dataset.PairwiseDataset import PairwiseDataset
import pandas as pd
from datasets import load_dataset

from autometrics.metrics.dummy import DummyMetric

# https://huggingface.co/datasets/nvidia/HelpSteer

class Design2Code(PairwiseDataset):
    def __init__(self, hf_path='SALT-NLP/Design2Code_human_eval_pairwise'):
        ds = load_dataset(hf_path)

        df = pd.DataFrame(ds['train'])

        target_columns_1 = ["win1"]
        target_columns_2 = ["win2"]
        ignore_columns = ["id", "ref_image", "ref_html", "model1", "model2", "image1", "image2", "html1", "html2", "win1", "win2", "tie"]
        metric_columns_1, metric_columns_2 = [], []

        name = "Design2Code"

        data_id_column = "id"
        model_id_column_1 = "model1"
        model_id_column_2 = "model2"
        input_column = "ref_html" # Technically should be ref_image, but we don't support images yet (TODO)
        output_column_1 = "html1"
        output_column_2 = "html2"
        reference_columns = ['ref_html']

        metrics = []

        super().__init__(dataframe=df, target_columns_1=target_columns_1, target_columns_2=target_columns_2,
                         ignore_columns=ignore_columns, metric_columns_1=metric_columns_1, metric_columns_2=metric_columns_2,
                         name=name, data_id_column=data_id_column, model_id_column_1=model_id_column_1, model_id_column_2=model_id_column_2,
                         input_column=input_column, output_column_1=output_column_1, output_column_2=output_column_2,
                         reference_columns=reference_columns, metrics=metrics)

