from autometrics.dataset.Dataset import Dataset
import pandas as pd
from autometrics.metrics.dummy import DummyMetric

class SummEval(Dataset):
    def __init__(self, path='./autometrics/dataset/datasets/summeval/model_annotations.aligned.paired.jsonl', include_precomputed=False):
        # Load the JSONL file into a DataFrame.
        df = pd.read_json(path, lines=True)

        # Compute averaged human scores for each of the four categories.
        def compute_avg_scores(row):
            # Concatenate expert and turker annotations.
            annotations = row['expert_annotations'] + row['turker_annotations']
            # Calculate averages for each score category.
            return pd.Series({
                'coherence': sum(a['coherence'] for a in annotations) / len(annotations),
                'consistency': sum(a['consistency'] for a in annotations) / len(annotations),
                'fluency': sum(a['fluency'] for a in annotations) / len(annotations),
                'relevance': sum(a['relevance'] for a in annotations) / len(annotations)
            })

        avg_scores = df.apply(compute_avg_scores, axis=1)
        df = pd.concat([df, avg_scores], axis=1)
        df.drop(columns=['expert_annotations', 'turker_annotations'], inplace=True)

        # Expand the 'references' column into separate columns: ref1, ref2, ...
        if 'references' in df.columns:
            # Convert the list of references into a DataFrame.
            references_df = pd.DataFrame(df['references'].tolist(), index=df.index)
            # Rename columns to ref1, ref2, ...
            references_df.rename(columns=lambda x: f"ref{x+1}", inplace=True)
            # Drop the original 'references' column and join the new ref columns.
            df = df.drop(columns=['references']).join(references_df)
            new_reference_columns = list(references_df.columns)
        else:
            new_reference_columns = []

        # Specify which columns are targets (the averaged human scores).
        target_columns = ['coherence', 'consistency', 'fluency', 'relevance']
        
        # Specify columns to ignore (metadata columns).
        ignore_columns = [
            "id", "model_id", "filepath", "decoded", "text"
        ]
        
        # The metric columns are those not in target, ignore, or reference columns.
        metric_columns = [col for col in df.columns 
                          if col not in target_columns 
                          and col not in ignore_columns 
                          and col not in new_reference_columns]
        
        if not include_precomputed:
            metric_columns = []
        
        # Define Dataset parameters.
        name = "SummEval"
        data_id_column = "id"
        model_id_column = "model_id"
        input_column = "text"
        output_column = "decoded"
        reference_columns = new_reference_columns

        # Instantiate dummy metrics (update later with real metrics as needed).
        metrics = [DummyMetric(col) for col in metric_columns]

        task_description = """Summarize the dialogue from the user conversation into a concise and coherent summary."""

        super().__init__(
            df, target_columns, ignore_columns, metric_columns, name,
            data_id_column, model_id_column, input_column, output_column,
            reference_columns, metrics, task_description=task_description
        )

if __name__ == "__main__":
    # Adjust the path to point to your JSONL file.
    dataset = SummEval(path='model_annotations.aligned.paired.jsonl')
    print(dataset.get_dataframe().head())