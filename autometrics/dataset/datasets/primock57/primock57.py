from autometrics.dataset.Dataset import Dataset
import pandas as pd

class Primock57(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)

        # add together incorrect_critical and incorrect_noncritical
        df['incorrect'] = df['incorrect_critical'] + df['incorrect_noncritical']

        # add together omissions_critical and omissions_noncritical
        df['omissions'] = df['omissions_critical'] + df['omissions_noncritical']

        # add together incorrect and omissions
        df['inc_plus_omi'] = df['incorrect'] + df['omissions']

        df.drop(columns=['incorrect_critical', 'incorrect_noncritical', 'omissions_critical', 'omissions_noncritical'], inplace=True)

        target_columns = ['time_sec', 'incorrect', 'omissions', 'inc_plus_omi']
        ignore_columns = ["consultation_id", "model_id", "evaluator", "generated_note", "human_note", "eval_note", "edited_note", "transcript"]
        metric_columns = [col for col in df.columns if col not in target_columns and col not in ignore_columns]

        name = "primock57"

        data_id_column = "consultation_id"
        model_id_column = "model_id"
        input_column = "transcript"
        output_column = "generated_note"
        reference_columns = ["human_note", "eval_note", "edited_note"]

        super().__init__(df, target_columns, ignore_columns, metric_columns, name, data_id_column, model_id_column, input_column, output_column, reference_columns)