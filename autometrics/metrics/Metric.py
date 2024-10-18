from abc import ABC, abstractmethod

class Metric(ABC):
    """
    Abstract class for metrics
    """
    def __init__(self, name, description):
        self.name = name
        self.description = description

    @abstractmethod
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

        return results

    @abstractmethod
    def predict(self, dataset, update_dataset=True, **kwargs):
        """
        Calculate the metric for the dataset
        """
        pass

    def get_name(self):
        return self.name
    
    def get_description(self):
        return self.description

    def __str__(self):
        return f"{self.name}: {self.description}"
    
    def __repr__(self):
        return self.__str__()