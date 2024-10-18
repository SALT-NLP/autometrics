from autometrics.metrics import Metric

class ReferenceFreeMetric(Metric):
    """
    Abstract class for reference-based metrics
    """
    def __init__(self, name, description):
        super().__init__(name, description)

    def calculate_row(self, row, dataset, update_dataset=True, **kwargs):
        """
        Calculate the metric
        """
        input_column = dataset.get_input_column()
        output_column = dataset.get_output_column()

        if not input_column:
            raise ValueError("Input column not found in dataset.  When constructing your Dataset please provide input_column.")
        if not output_column:
            raise ValueError("Output column not found in dataset.  When constructing your Dataset please provide output_column.")

        input = row[input_column]
        output = row[output_column]

        result = self.calculate(input, output, **kwargs)

        if update_dataset:
            row[self.name] = result

        return result

    def predict(self, dataset, update_dataset=True, **kwargs):
        """
        Calculate the metric for the dataset
        """
        df = dataset.get_dataframe()
        input_column = dataset.get_input_column()
        output_column = dataset.get_output_column()

        if not input_column:
            raise ValueError("Input column not found in dataset.  When constructing your Dataset please provide input_column.")
        if not output_column:
            raise ValueError("Output column not found in dataset.  When constructing your Dataset please provide output_column.")

        inputs = df[input_column].values.tolist()
        outputs = df[output_column].values.tolist()

        results = self.calculate_batched(inputs, outputs)

        if update_dataset:
            df[self.name] = results
            dataset.set_dataframe(df)

            if self.name not in dataset.get_metric_columns():
                dataset.get_metric_columns().append(self.name)

        return results

