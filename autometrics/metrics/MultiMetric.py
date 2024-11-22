from autometrics.metrics.Metric import Metric

class MultiMetric(Metric):
    """
    Abstract class for metrics
    """
    def __init__(self, name, description, submetric_names=[]) -> None:
        self.name = name
        self.description = description
        self.submetric_names = submetric_names

    def calculate(self, input, output, references=None, **kwargs):
        """
        Calculate the metric
        """
        pass
        

    def calculate_batched(self, inputs, outputs, references=None, **kwargs):
        """
        Calculate the metric for a batch of inputs and outputs. The default implementation simply calls calculate for each input/output pair. Override this method if you can calculate the metric more efficiently for a batch of inputs/outputs.
        """
        if references is None:
            references = [None] * len(inputs)

        results = []
        for i, o, r in zip(inputs, outputs, references):
            results.append(self.calculate(i, o, r, **kwargs))

        # Swap the indices so that each submetric has its own list
        results = list(zip(*results))

        return results

    def predict(self, dataset, update_dataset=True, **kwargs):
        """
        Calculate the metric for the dataset
        """
        pass

    def get_submetric_names(self):
        return self.submetric_names