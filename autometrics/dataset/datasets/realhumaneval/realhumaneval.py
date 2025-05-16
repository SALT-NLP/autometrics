from autometrics.dataset.Dataset import Dataset
import pandas as pd
from datasets import load_dataset

from autometrics.metrics.dummy import DummyMetric

# https://huggingface.co/datasets/hsseinmz/realhumaneval/viewer/autocomplete/train

class RealHumanEval(Dataset):
    def __init__(self, hf_path='hsseinmz/realhumaneval'):
        ds = load_dataset(hf_path, 'autocomplete')

        df = pd.DataFrame(ds['train'])

        df['input'] = df.apply(lambda x: f"{x['prefix_code']}[AI GENERATED CODE GOES HERE]{x['suffix_code']}", axis=1)
        
        # Drop columns with no suggestions since accepting or rejecting them is equivalent
        df = df[df['suggestion'].notna()]
        df = df[df['suggestion'] != '']

        target_columns = ['accepted']
        ignore_columns = ["prefix_code","suffix_code","logprobs","accepted","suggestion","programmer_id","timestamp","model","task_name","requested"]
        metric_columns = [col for col in df.columns if col not in target_columns and col not in ignore_columns]

        name = "RealHumanEval"

        data_id_column = "timestamp"
        model_id_column = "model"
        input_column = "input"
        output_column = "suggestion"
        reference_columns = []

        metrics = [DummyMetric(col) for col in metric_columns]

        task_description = """You are an expert Python programmer, be helpful to the user and return code only in Python."""

        super().__init__(df, target_columns, ignore_columns, metric_columns, name, data_id_column, model_id_column, input_column, output_column, reference_columns, metrics, task_description=task_description)
    