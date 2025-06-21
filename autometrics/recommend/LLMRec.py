import dspy
from typing import List, Type

from autometrics.metrics.Metric import Metric
from autometrics.dataset.Dataset import Dataset
from autometrics.recommend.MetricRecommender import MetricRecommender
from autometrics.recommend.utils import metric_name_to_class


class LLMMetricRecommendationSignature(dspy.Signature):
    """I am looking for a metric to evaluate the attached task. In particular I care about the specific target measurement that I attached.

Please help me decide from among the metrics that I have attached documentation for which one is most relevant to the task and target.

Please provide a ranking of the metrics from most relevant to least relevant for the task and target above.
You can reason first about what makes a metric relevant for the task and target, and then provide your ranking.

The final ranking should just be a list of metric names, in order from most relevant to least relevant.

You can include as much reasoning as you need, but please make your final answer a single list of metric names, in order from most relevant to least relevant."""
    task_description: str = dspy.InputField(desc="A description of the task that an LLM performed and that I now want to evaluate.")
    target: str = dspy.InputField(desc="The specific target measurement that I want to evaluate about the task.")
    num_metrics_to_recommend: int = dspy.InputField(desc="The number of metrics to recommend.")
    metric_documentation: List[str] = dspy.InputField(desc="A list of metric names and their documentation.  The documentation will contain the metric name, as well as many details about the metric.")
    ranking: List[str] = dspy.OutputField(desc="A list of metric names, in order from most relevant to least relevant.  The list should be of length `num_metrics_to_recommend`.")

class LLMMetricRecommendation(dspy.Module):
    """
    A module that recommends metrics for a given task and target.
    """
    def __init__(self):
        super().__init__()
        self.recommender = dspy.ChainOfThought(LLMMetricRecommendationSignature)

    def forward(self, task_description: str, target: str, num_metrics_to_recommend: int, metric_documentation: List[str]) -> List[str]:
        results = self.recommender(task_description=task_description, target=target, num_metrics_to_recommend=num_metrics_to_recommend, metric_documentation=metric_documentation)
        return results.ranking
    
class LLMRec(MetricRecommender):
    """
    A metric recommender that uses a LLM to recommend metrics.
    """
    def __init__(self, metric_classes: List[Type[Metric]], index_path: str, force_reindex: bool = False, model: dspy.LM = None):
        super().__init__(metric_classes, index_path, force_reindex)
        self.model = model
        self.recommender = LLMMetricRecommendation()

    def recommend(self, dataset: Dataset, target_measurement: str, k: int = 20) -> List[Type[Metric]]:
        task_description = dataset.get_task_description()
        metric_documentation = [f"«METRIC NAME: {metric.__name__}\nMETRIC DOCUMENTATION: {metric.__doc__}»" for metric in self.metric_classes]

        if self.model is not None:
            with dspy.settings.context(lm = self.model):
                ranking = self.recommender(task_description, target_measurement, k, metric_documentation)
        else:
            ranking = self.recommender(task_description, target_measurement, k, metric_documentation)
        results = [metric_name_to_class(metric_name) for metric_name in ranking]
        if len(results) > k:
            results = results[:k]
        return results