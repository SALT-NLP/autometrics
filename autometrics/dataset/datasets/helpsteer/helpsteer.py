from autometrics.dataset.Dataset import Dataset
import pandas as pd
from datasets import load_dataset

from autometrics.metrics.dummy import DummyMetric

# https://huggingface.co/datasets/nvidia/HelpSteer

class HelpSteer(Dataset):
    def __init__(self, hf_path='nvidia/HelpSteer'):
        ds = load_dataset(hf_path)

        df = pd.DataFrame(ds['train'])

        df['id'] = df.apply(lambda x: f"{hash(x['prompt'])}", axis=1)

        target_columns = ['helpfulness','correctness','coherence','complexity','verbosity']
        ignore_columns = ["id","prompt","response"]
        metric_columns = [col for col in df.columns if col not in target_columns and col not in ignore_columns]

        name = "HelpSteer"

        data_id_column = "id"
        model_id_column = None
        input_column = "prompt"
        output_column = "response"
        reference_columns = []

        metrics = [DummyMetric(col) for col in metric_columns]

        train_df = df.copy()
        val_df = pd.DataFrame(ds['validation'])
        val_df['id'] = val_df.apply(lambda x: f"{hash(x['prompt'])}", axis=1)

        self.train_dataset = Dataset(train_df, target_columns, ignore_columns, metric_columns, name, data_id_column, model_id_column, input_column, output_column, reference_columns, metrics)
        self.val_dataset = Dataset(val_df, target_columns, ignore_columns, metric_columns, name, data_id_column, model_id_column, input_column, output_column, reference_columns, metrics)
        self.test_dataset = None
        task_description = """Answer the user query as a helpful chatbot assistant."""

        super().__init__(df, target_columns, ignore_columns, metric_columns, name, data_id_column, model_id_column, input_column, output_column, reference_columns, metrics, task_description=task_description)

    def get_splits(self, split_column=None, train_ratio=0.5, val_ratio=0.2, seed=None, preserve_splits=True, max_size=None):
        if preserve_splits:
            # The validation set is best used as the test set since we don't have a test set
            updated_train_ratio = (train_ratio) / (train_ratio + val_ratio)
            updated_val_ratio = (val_ratio) / (train_ratio + val_ratio)
            trainset, valset, _ = self.train_dataset.get_splits(split_column, updated_train_ratio, updated_val_ratio, seed)
            if max_size:
                trainset = trainset.get_subset(max_size)
                valset = valset.get_subset(max_size)
            return trainset, valset, self.val_dataset

        else:
            return super().get_splits(split_column, train_ratio, val_ratio, seed, max_size=max_size)

    def get_kfold_splits(self, k=5, split_column=None, seed=None, test_ratio=0.3, preserve_splits=True):
        if preserve_splits:
            splits, train_dataset, _ = self.train_dataset.get_kfold_splits(k, split_column, seed, 0.0)
            test_dataset = self.val_dataset
            return splits, train_dataset, test_dataset
        else:
            return super().get_kfold_splits(k, split_column, seed, test_ratio)
        
class HelpSteer2(HelpSteer):
    def __init__(self):
        super().__init__('nvidia/HelpSteer2')


if __name__ == "__main__":
    HelpSteer2()

