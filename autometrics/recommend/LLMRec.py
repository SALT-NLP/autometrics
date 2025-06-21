import dspy
from typing import List, Type
import math

from autometrics.metrics.Metric import Metric
from autometrics.dataset.Dataset import Dataset
from autometrics.recommend.MetricRecommender import MetricRecommender
from autometrics.recommend.utils import metric_name_to_class


class LLMMetricRecommendationSignature(dspy.Signature):
    """I am looking for a metric to evaluate the attached task. In particular I care about the specific target measurement that I attached.

Please help me decide from among the metrics that I have attached documentation for which one is most relevant to the task and target.

Please provide a ranking of the metrics from most relevant to least relevant for the task and target above.
You can reason first about what makes a metric relevant for the task and target, and then provide your ranking.

IMPORTANT: The final ranking should be a list of EXACT metric class names (no hyphens, no spaces, no extra words).  Use the METRIC NAME not what it is called in the documentation.
For example, use "SelfBLEU" not "Self-BLEU", use "BERTScore" not "BERT Score", use "BLEU" not "BLEU Score".

The final ranking should just be a list of metric names, in order from most relevant to least relevant.
The list should be exactly `num_metrics_to_recommend` items long."""
    task_description: str = dspy.InputField(desc="A description of the task that an LLM performed and that I now want to evaluate.")
    target: str = dspy.InputField(desc="The specific target measurement that I want to evaluate about the task.")
    num_metrics_to_recommend: int = dspy.InputField(desc="The number of metrics to recommend.")
    metric_documentation: List[str] = dspy.InputField(desc="A list of metric names and their documentation.  The documentation will contain the metric name, as well as many details about the metric.")
    ranking: List[str] = dspy.OutputField(desc="A numbered list of EXACT metric class names (no hyphens, no spaces, no extra words), in order from most relevant to least relevant. The list should be of length `num_metrics_to_recommend`.  You should write the number in front of the metric name (e.g \"1. METRIC1_NAME\", \"2. METRIC2_NAME\", etc.)")

class LLMMetricRecommendation(dspy.Module):
    """
    A module that recommends metrics for a given task and target.
    """
    def __init__(self):
        super().__init__()
        self.recommender = dspy.ChainOfThought(LLMMetricRecommendationSignature)

    def forward(self, task_description: str, target: str, num_metrics_to_recommend: int, metric_documentation: List[str]) -> List[str]:
        results = self.recommender(task_description=task_description, target=target, num_metrics_to_recommend=num_metrics_to_recommend, metric_documentation=metric_documentation)

        # Remove the numbers from the ranking, accounting for the fact that there may be no numbers in the ranking
        results.ranking = [rank.split(".")[1].strip() if "." in rank else rank.strip() for rank in results.ranking]
        
        return results.ranking
    
class LLMRec(MetricRecommender):
    """
    A metric recommender that uses a LLM to recommend metrics.
    """
    def __init__(self, metric_classes: List[Type[Metric]], index_path: str = None, force_reindex: bool = False, model: dspy.LM = None):
        super().__init__(metric_classes, index_path, force_reindex)
        self.model = model
        self.recommender = LLMMetricRecommendation()

    def _recommend_batch(self, metric_classes: List[Type[Metric]], dataset: Dataset, target_measurement: str, k: int) -> List[str]:
        """
        Helper method to recommend metrics from a batch of metric classes.
        Returns the raw ranking (list of metric names) rather than metric classes.
        """
        task_description = dataset.get_task_description()
        metric_documentation = [f"«METRIC NAME: {metric.__name__}\nMETRIC DOCUMENTATION: {metric.__doc__}»" for metric in metric_classes]

        if self.model is not None:
            with dspy.settings.context(lm=self.model):
                ranking = self.recommender(task_description, target_measurement, k, metric_documentation)
        else:
            ranking = self.recommender(task_description, target_measurement, k, metric_documentation)
        
        return ranking

    def _split_metrics_into_batches(self, metric_classes: List[Type[Metric]], num_batches: int) -> List[List[Type[Metric]]]:
        """
        Split metric classes into approximately equal batches.
        """
        batch_size = math.ceil(len(metric_classes) / num_batches)
        batches = []
        for i in range(0, len(metric_classes), batch_size):
            batches.append(metric_classes[i:min(i + batch_size, len(metric_classes))])
        return batches

    def _recommend_with_splitting(self, metric_classes: List[Type[Metric]], dataset: Dataset, target_measurement: str, k: int, max_depth: int = 5) -> List[Type[Metric]]:
        """
        Recursively recommend metrics, splitting batches when context window is exceeded.
        
        Args:
            metric_classes: List of metric classes to recommend from
            dataset: Dataset for task description
            target_measurement: Target measurement to optimize for
            k: Number of metrics to recommend
            max_depth: Maximum recursion depth to prevent infinite recursion
            
        Returns:
            List of recommended metric classes
        """
        if max_depth <= 0:
            raise RuntimeError("Maximum recursion depth reached in metric recommendation. Cannot handle context window with current metric set.")
        
        # If we have very few metrics, try direct recommendation
        if len(metric_classes) <= 3:
            try:
                ranking = self._recommend_batch(metric_classes, dataset, target_measurement, min(k, len(metric_classes)))
                results = [metric_name_to_class(metric_name) for metric_name in ranking]
                return results[:k] if len(results) > k else results
            except Exception as e:
                if "ContextWindowExceededError" in str(e):
                    # If we still can't handle it with very few metrics, we have a fundamental problem
                    raise RuntimeError(f"Context window exceeded even with {len(metric_classes)} metrics. Individual metric documentation may be too long.")
                else:
                    raise e

        try:
            # Try to recommend directly from all metrics
            ranking = self._recommend_batch(metric_classes, dataset, target_measurement, k)
            results = [metric_name_to_class(metric_name) for metric_name in ranking]
            return results[:k] if len(results) > k else results
            
        except Exception as e:
            if "ContextWindowExceededError" not in str(e):
                # If it's not a context window error, re-raise
                raise e
            
            # Context window exceeded, split into batches
            print(f"Context window exceeded with {len(metric_classes)} metrics. Splitting into batches...")
            
            # Split into 2 batches (could be made configurable)
            batches = self._split_metrics_into_batches(metric_classes, 2)
            
            # Calculate target k for each batch (aim for ~2/3 of final k to allow for overlap)
            batch_k = max(1, int(k * 0.67))  # Ensure at least 1
            
            # Recursively get recommendations from each batch
            all_batch_results = []
            for i, batch in enumerate(batches):
                print(f"Processing batch {i+1}/{len(batches)} with {len(batch)} metrics, targeting {batch_k} recommendations...")
                batch_results = self._recommend_with_splitting(batch, dataset, target_measurement, batch_k, max_depth - 1)
                all_batch_results.extend(batch_results)
            
            # Remove duplicates while preserving order
            seen = set()
            merged_results = []
            for metric in all_batch_results:
                if metric not in seen:
                    merged_results.append(metric)
                    seen.add(metric)
            
            print(f"Merged {len(merged_results)} unique recommendations from batches. Performing final ranking...")
            
            # If we have fewer or equal results than k, return them directly
            if len(merged_results) <= k:
                return merged_results
            
            # Final recommendation pass on the merged results to get the top k
            return self._recommend_with_splitting(merged_results, dataset, target_measurement, k, max_depth - 1)

    def recommend(self, dataset: Dataset, target_measurement: str, k: int = 20) -> List[Type[Metric]]:
        """
        Recommend metrics for a given dataset and target measurement.
        Handles context window exceeded errors by recursively splitting metric batches.
        """
        return self._recommend_with_splitting(self.metric_classes, dataset, target_measurement, k)