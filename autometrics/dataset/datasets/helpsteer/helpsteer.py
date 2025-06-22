from autometrics.dataset.Dataset import Dataset
import pandas as pd
from datasets import load_dataset

from autometrics.metrics.dummy import DummyMetric

# https://huggingface.co/datasets/nvidia/HelpSteer

class HelpSteer(Dataset):
    def __init__(self, hf_path='nvidia/HelpSteer'):
        ds = load_dataset(hf_path)

        df = pd.DataFrame(ds['train'])
        val_df = pd.DataFrame(ds['validation'])

        # Find and remove overlapping prompts between train and validation sets
        train_prompts = set(df['prompt'])
        val_prompts = set(val_df['prompt'])
        overlapping_prompts = train_prompts & val_prompts
        
        if overlapping_prompts:
            print(f"Warning: Found {len(overlapping_prompts)} overlapping prompts between train and validation sets. Removing from training set.")
            # Remove overlapping prompts from the training set (keep them in validation)
            df = df[~df['prompt'].isin(overlapping_prompts)].copy()

        df['id'] = df.apply(lambda x: f"{hash(x['prompt'])}", axis=1)
        val_df['id'] = val_df.apply(lambda x: f"{hash(x['prompt'])}", axis=1)

        target_columns = ['helpfulness','correctness','coherence','complexity','verbosity']
        ignore_columns = ["id","prompt","response"]
        metric_columns = [col for col in df.columns if col not in target_columns and col not in ignore_columns]

        # Derive name from the dataset path
        name = "HelpSteer" if hf_path == 'nvidia/HelpSteer' else "HelpSteer2"

        data_id_column = "id"
        model_id_column = None
        input_column = "prompt"
        output_column = "response"
        reference_columns = []

        metrics = [DummyMetric(col) for col in metric_columns]

        train_df = df.copy()

        self.train_dataset = Dataset(train_df, target_columns, ignore_columns, metric_columns, name, data_id_column, model_id_column, input_column, output_column, reference_columns, metrics)
        self.val_dataset = Dataset(val_df, target_columns, ignore_columns, metric_columns, name, data_id_column, model_id_column, input_column, output_column, reference_columns, metrics)
        self.test_dataset = None
        task_description = """Answer the user query as a helpful chatbot assistant."""

        super().__init__(df, target_columns, ignore_columns, metric_columns, name, data_id_column, model_id_column, input_column, output_column, reference_columns, metrics, task_description=task_description)

    def get_splits(self, split_column=None, train_ratio=0.5, val_ratio=0.2, seed=None, max_size=None, *, preserve_splits=True):
        if preserve_splits:
            # Use the validation set as the test set (consistent with the intent)
            test_dataset = self.val_dataset
            
            # Apply max_size to test set FIRST to ensure consistency with k-fold splits
            if max_size:
                test_dataset = test_dataset.get_subset(max_size, seed=42)
            
            # Now split the training portion using our consistent logic
            # Calculate adjusted ratios for splitting the training data only
            total_train_val_ratio = train_ratio + val_ratio  
            adjusted_train_ratio = train_ratio / total_train_val_ratio
            adjusted_val_ratio = val_ratio / total_train_val_ratio
            
            # Split the training dataset using our unified logic
            train_dataset, val_dataset, _ = self.train_dataset.get_splits(
                split_column=split_column, 
                train_ratio=adjusted_train_ratio, 
                val_ratio=adjusted_val_ratio, 
                seed=seed
            )
            
            if max_size:
                train_dataset = train_dataset.get_subset(max_size, seed=42)
                val_dataset = val_dataset.get_subset(max_size, seed=42)
                
            return train_dataset, val_dataset, test_dataset
        else:
            return super().get_splits(split_column, train_ratio, val_ratio, seed, max_size=max_size)

    def get_kfold_splits(self, k=5, split_column=None, seed=None, test_ratio=0.3, max_size=None, *, preserve_splits=True):
        if preserve_splits:
            # Use the validation set as the test set (consistent with get_splits)
            test_dataset = self.val_dataset
            
            # Apply max_size to test set FIRST to ensure consistency with regular splits
            if max_size:
                test_dataset = test_dataset.get_subset(max_size, seed=42)
            
            # Create k-fold splits from the training data only (no separate test extraction)
            splits, train_dataset, _ = self.train_dataset.get_kfold_splits(
                k=k, 
                split_column=split_column, 
                seed=seed, 
                test_ratio=0.0,  # No test extraction from training data
                max_size=max_size
            )
            
            return splits, train_dataset, test_dataset
        else:
            return super().get_kfold_splits(k, split_column, seed, test_ratio, max_size)
        
class HelpSteer2(HelpSteer):
    def __init__(self):
        super().__init__('nvidia/HelpSteer2')


if __name__ == "__main__":
    HelpSteer2()

