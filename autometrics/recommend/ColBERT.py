from typing import List, Type, Optional
from autometrics.metrics.Metric import Metric
from autometrics.dataset.Dataset import Dataset
from autometrics.recommend.MetricRecommender import MetricRecommender
from autometrics.recommend.utils import metric_name_to_class

import os
from platformdirs import user_data_dir
from ragatouille import RAGPretrainedModel


class ColBERT(MetricRecommender):
    """Metric recommender that leverages ColBERT-v2 via the *ragatouille* package.

    The first time the class is instantiated (or when *force_reindex* is ``True``)
    we create a ColBERT index over the provided ``metric_classes``.  Subsequent
    instantiations simply load the existing index which keeps start-up time low.
    """

    def __init__(
        self,
        metric_classes: List[Type[Metric]],
        index_path: Optional[str] = user_data_dir("autometrics", "colbert"),
        force_reindex: bool = False,
        index_name: str = "all_metrics",
    ) -> None:
        # Store the metric classes that will be indexed/searched
        self.metric_classes: List[Type[Metric]] = metric_classes

        self.index_path: str = index_path
        self.force_reindex: bool = force_reindex
        self.index_name: str = index_name

        # If the index is missing (or the user explicitly asked for a rebuild)
        # we need to construct it from scratch.
        if force_reindex or not os.path.exists(self.index_path):
            self._build_index()
        elif not self.index_name in self.index_path:
            self.index_path = os.path.join(self.index_path, "colbert", "indexes", self.index_name)

        # Finally, load the search object
        self.rag = RAGPretrainedModel.from_index(self.index_path)

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _build_index(self) -> None:
        """Create a ColBERT index for *metric_classes* at *index_path*."""
        # Make sure the parent directory exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        # Prepare the documents: one document per metric class consisting of its
        # docstring.  If a docstring is missing we fall back to an empty string
        # to keep the ordering consistent.
        metric_ids: List[str] = [mc.__name__ for mc in self.metric_classes]
        metric_docs: List[str] = [(mc.__doc__ or "") for mc in self.metric_classes]

        print(
            f"Building ColBERT index at {self.index_path} for {len(metric_ids)} metrics..."
        )

        rag_builder = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0", index_path=self.index_path)
        built_path: str = rag_builder.index(
            index_name=self.index_name,
            collection=metric_docs,
            document_ids=metric_ids,
        )

        # Update instance path to the freshly built location
        self.index_path = built_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def recommend(
        self, dataset: Dataset, target_measurement: str, k: int = 20
    ) -> List[Type[Metric]]:
        """Return the *k* most relevant metrics for *(dataset, target_measurement)*.

        The query template follows the pattern we established for the BM25
        recommender so that both systems are directly comparable.
        """
        # Robustly obtain a human-readable task description from the dataset
        task_desc = dataset.get_task_description()

        task_desc = task_desc or dataset.get_name() if hasattr(dataset, "get_name") else ""

        query = (
            f'I am looking for a metric to evaluate the following task: "{task_desc}" '
            f' In particular I care about "{target_measurement}".'
        )
        results = self.rag.search(query, k=k)
        return [metric_name_to_class(hit["document_id"]) for hit in results]
