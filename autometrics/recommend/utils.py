from typing import Type
from autometrics.metrics.MetricBank import all_metric_classes

metric_map = { metric_class.__name__: metric_class for metric_class in all_metric_classes }

def metric_name_to_class(metric_name: str) -> Type[Metric]:
    return metric_map[metric_name]