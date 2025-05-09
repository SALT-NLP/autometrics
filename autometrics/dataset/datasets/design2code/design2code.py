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
        task_description = """You are an expert web developer who specializes in HTML and CSS. A user will provide you with a screenshot of a webpage. You need to return a single html file that uses HTML and CSS to reproduce the given website. Include all CSS code in the HTML file itself. If it involves any images, use "rick.jpg" as the placeholder. Some images on the webpage are replaced with a blue rectangle as the placeholder, use "rick.jpg" for those as well. Do not hallucinate any dependencies to external files. You do not need to include JavaScript scripts for dynamic interactions. Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout. Respond with the content of the HTML+CSS file."""

        metrics = []

        super().__init__(dataframe=df, target_columns_1=target_columns_1, target_columns_2=target_columns_2,
                         ignore_columns=ignore_columns, metric_columns_1=metric_columns_1, metric_columns_2=metric_columns_2,
                         name=name, data_id_column=data_id_column, model_id_column_1=model_id_column_1, model_id_column_2=model_id_column_2,
                         input_column=input_column, output_column_1=output_column_1, output_column_2=output_column_2,
                         reference_columns=reference_columns, metrics=metrics, task_description=task_description)

