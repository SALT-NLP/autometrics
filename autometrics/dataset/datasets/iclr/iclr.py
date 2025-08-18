from autometrics.dataset.Dataset import Dataset
import pandas as pd
from autometrics.metrics.dummy import DummyMetric


class ICLR(Dataset):
    def __init__(self, dir_path: str = './autometrics/dataset/datasets/iclr'):
        # Load provided splits
        train_path = f"{dir_path}/train.csv"
        dev_path = f"{dir_path}/dev.csv"
        test_path = f"{dir_path}/test.csv"

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(dev_path)
        test_df = pd.read_csv(test_path)

        # Column roles
        # Identifiers and metadata
        data_id_column = 'id'
        model_id_column = None

        # Text fields
        input_column = 'abstract'
        output_column = 'full_text'
        reference_columns = []

        # Targets: review aspect scores, overall recommendation, and acceptance label
        target_columns = [
            'recommendation',
            'appropriateness',
            'clarity',
            'impact',
            'meaningful_comparison',
            'originality',
            'replicability',
            'reviewer_confidence',
            'soundness_correctness',
            'substance',
            'accepted',
        ]

        # Columns to ignore for metric computation
        ignore_columns = [
            'id',
            'full_text',
            'conference',
            'format',
            'review_comments',
            'abstract',
        ]

        # Optional auxiliary numeric/text columns that could be tracked as metrics if present
        # Exclude targets and ignored columns
        metric_columns = [
            col for col in train_df.columns
            if col not in target_columns and col not in ignore_columns
        ]

        name = 'iclr'

        # Instantiate metric objects (no-op placeholders by default)
        metrics = [DummyMetric(col) for col in metric_columns]

        # Base dataset will mirror the pattern used elsewhere: use training as the base
        task_description = (
            "Write a high quality ICLR paper on a specific topic.  Sometimes the abstract will be provided as scaffolding."
        )

        # Store preserved split datasets
        self.train_dataset = Dataset(
            train_df.copy(), target_columns, ignore_columns, metric_columns, name,
            data_id_column, model_id_column, input_column, output_column,
            reference_columns, metrics, task_description=task_description
        )
        self.val_dataset = Dataset(
            val_df.copy(), target_columns, ignore_columns, metric_columns, name,
            data_id_column, model_id_column, input_column, output_column,
            reference_columns, metrics, task_description=task_description
        )
        self.test_dataset = Dataset(
            test_df.copy(), target_columns, ignore_columns, metric_columns, name,
            data_id_column, model_id_column, input_column, output_column,
            reference_columns, metrics, task_description=task_description
        )

        # Initialize parent with training dataframe (consistent with other loaders)
        super().__init__(
            train_df.copy(), target_columns, ignore_columns, metric_columns, name,
            data_id_column, model_id_column, input_column, output_column,
            reference_columns, metrics, task_description=task_description
        )

    def get_splits(self, split_column=None, train_ratio=0.5, val_ratio=0.2, seed=None, max_size=None, *, preserve_splits=True):
        if preserve_splits:
            train_dataset = self.train_dataset
            val_dataset = self.val_dataset
            test_dataset = self.test_dataset

            if max_size:
                # Apply max_size consistently to each split
                train_dataset = train_dataset.get_subset(max_size, seed=42 if seed is None else seed)
                val_dataset = val_dataset.get_subset(max_size, seed=42 if seed is None else seed)
                test_dataset = test_dataset.get_subset(max_size, seed=42 if seed is None else seed)

            return train_dataset, val_dataset, test_dataset
        else:
            return super().get_splits(split_column, train_ratio, val_ratio, seed, max_size=max_size)

    def get_kfold_splits(self, k=5, split_column=None, seed=None, test_ratio=0.3, max_size=None, *, preserve_splits=True):
        if preserve_splits:
            # Keep provided test set fixed; create folds from training data only
            test_dataset = self.test_dataset
            if max_size:
                test_dataset = test_dataset.get_subset(max_size, seed=42 if seed is None else seed)

            splits, train_dataset, _ = self.train_dataset.get_kfold_splits(
                k=k,
                split_column=split_column,
                seed=seed,
                test_ratio=0.0,
                max_size=max_size,
            )
            return splits, train_dataset, test_dataset
        else:
            return super().get_kfold_splits(k, split_column, seed, test_ratio, max_size)


if __name__ == "__main__":
    ICLR()


