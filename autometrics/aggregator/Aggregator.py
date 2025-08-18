from abc import ABC, abstractmethod
from autometrics.metrics.MultiMetric import MultiMetric
from typing import List

from autometrics.util.metric_eval_utils import evaluate_metric_instances

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
        if not self.input_metrics:
            return

        # Work off the actual DataFrame columns to avoid stale metadata in metric_columns
        df = dataset.get_dataframe()

        # Collect metrics that are missing and require computation
        metrics_to_compute: List = []
        for metric in self.input_metrics:
            if isinstance(metric, MultiMetric):
                submetric_names = metric.get_submetric_names()
                missing = [name for name in submetric_names if name not in df.columns]
                if missing:
                    metrics_to_compute.append(metric)
            else:
                metric_name = metric.get_name()
                if metric_name not in df.columns:
                    metrics_to_compute.append(metric)

        # If nothing to compute, just ensure metadata is synced
        if not metrics_to_compute:
            for metric in self.input_metrics:
                if isinstance(metric, MultiMetric):
                    for name in metric.get_submetric_names():
                        if name not in dataset.get_metric_columns():
                            dataset.get_metric_columns().append(name)
                else:
                    metric_name = metric.get_name()
                    if metric_name not in dataset.get_metric_columns():
                        dataset.get_metric_columns().append(metric_name)
            return

        evaluated = evaluate_metric_instances(
            dataset=dataset,
            metric_instances=metrics_to_compute,
            enable_parallel=True,
            max_parallel_workers=20,
            allowed_failed_metrics=0,
        )
        # Sync metadata for evaluated metrics
        df = dataset.get_dataframe()
        for metric in evaluated:
            if isinstance(metric, MultiMetric):
                for name in metric.get_submetric_names():
                    if name in df.columns and name not in dataset.get_metric_columns():
                        dataset.get_metric_columns().append(name)
            else:
                metric_name = metric.get_name()
                if metric_name in df.columns and metric_name not in dataset.get_metric_columns():
                    dataset.get_metric_columns().append(metric_name)

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