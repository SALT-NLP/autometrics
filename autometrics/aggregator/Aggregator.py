from abc import ABC, abstractmethod
from autometrics.metrics.MultiMetric import MultiMetric

class Aggregator(ABC):
    """
    Abstract class for metric aggregator
    """
    def __init__(self, name, description, input_metrics=None, dataset=None, **kwargs):
        self.name = name
        self.description = description
        if input_metrics is None:
            self.input_metrics = dataset.get_metrics()
        else:
            self.input_metrics = input_metrics
        self.dataset = dataset

    def ensure_dependencies(self, dataset):
        """
        Ensure that the input metrics are present in the dataset
        """
        if self.input_metrics:
            for metric in self.input_metrics:
                if isinstance(metric, MultiMetric):
                    for submetric in metric.get_submetric_names():
                        if submetric not in dataset.get_metric_columns():
                            metric.predict(dataset)
                elif metric.get_name() not in dataset.get_metric_columns():
                    metric.predict(dataset)

                df = dataset.get_dataframe()

                for i, row in df.iterrows():
                    if isinstance(metric, MultiMetric):
                        for submetric in metric.get_submetric_names():
                            if submetric not in row:
                                metric.calculate_row(row, dataset, update_dataset=True)
                    elif metric.get_name() not in row:
                        metric.calculate_row(row, dataset, update_dataset=True)

                dataset.set_dataframe(df)

    def get_input_columns(self):
        """
        Get the input columns
        """
        input_cols = []
        for metric in self.input_metrics:
            if isinstance(metric, MultiMetric):
                input_cols.extend(metric.get_submetric_names())
            else:
                input_cols.append(metric.get_name())
        return input_cols

    @abstractmethod
    def _predict_unsafe(self, dataset, update_dataset=True):
        """
        Calculate the aggregation without ensuring dependencies
        """
        pass

    def predict(self, dataset, update_dataset=True):
        """
        Calculate the aggregation
        """
        self.ensure_dependencies(dataset)
        return self._predict_unsafe(dataset, update_dataset)

    def __str__(self):
        return f"{self.name}: {self.description}"
    
    def __repr__(self):
        return self.__str__()
    
    @abstractmethod
    def identify_important_metrics(self):
        """
        Identify the important metrics
        """
        pass

    def get_name(self):
        return self.name
    
    def get_description(self):
        return self.description