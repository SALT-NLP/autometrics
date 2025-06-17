from abc import ABC, abstractmethod

from typing import List, Type
from autometrics.metrics.base import Metric
from autometrics.datasets.base import Dataset


class MetricRecommender(ABC):
    def __init__(self, metric_classes: List[Type[Metric]], index_path: str, force_reindex: bool = False):
        self.metric_classes = metric_classes
        self.index_path = index_path
        self.force_reindex = force_reindex

    @abstractmethod
    def recommend(self, dataset: Dataset, target_measurement: str, k: int = 20) -> List[Type[Metric]]:
        """
        Recommend metrics for a given dataset.

        Args:
            dataset: The dataset to recommend metrics for.
            target_measurement: The target measurement to recommend metrics for.
            k: The number of metrics to recommend.
        Returns:
            An ordered list of recommended metrics (first is most recommended)
        """
        pass