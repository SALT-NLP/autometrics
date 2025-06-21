from typing import List, Type, Optional
from autometrics.metrics.Metric import Metric
from autometrics.dataset.Dataset import Dataset
from autometrics.recommend.MetricRecommender import MetricRecommender
from autometrics.recommend.utils import metric_name_to_class

import os
from platformdirs import user_data_dir
from pylate import indexes, models, retrieve


class ColBERT(MetricRecommender):
    """Metric recommender that leverages Modern ColBERT via the PyLate package.

    The first time the class is instantiated (or when *force_reindex* is ``True``)
    we create a ColBERT index over the provided ``metric_classes``.  Subsequent
    instantiations simply load the existing index which keeps start-up time low.
    """

    def __init__(
        self,
        metric_classes: List[Type[Metric]],
        index_path: Optional[str] = user_data_dir("autometrics", "colbert"),
        force_reindex: bool = False,
    ) -> None:
        # Store the metric classes that will be indexed/searched
        self.metric_classes: List[Type[Metric]] = metric_classes

        self.index_path: str = index_path
        self.force_reindex: bool = force_reindex
        self.index_name: str = "colbert_index" # Causes too many issues when this is a parameter.  Just use the index path to determine the index name.

        # Initialize the ColBERT model
        self.model = models.ColBERT(
            model_name_or_path="lightonai/GTE-ModernColBERT-v1",
        )

        # If the index is missing (or the user explicitly asked for a rebuild)
        # we need to construct it from scratch.
        if force_reindex or not os.path.exists(self.index_path):
            self._build_index()
        
        # Initialize the PLAID index
        self.index = indexes.PLAID(
            index_folder=self.index_path,
            index_name=self.index_name,
            override=False,
        )
        
        # Initialize the retriever
        self.retriever = retrieve.ColBERT(index=self.index)

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

        # Encode the documents
        documents_embeddings = self.model.encode(
            metric_docs,
            batch_size=32,
            is_query=False,
            show_progress_bar=True,
        )

        # Create and initialize the PLAID index
        index = indexes.PLAID(
            index_folder=self.index_path,
            index_name=self.index_name,
            override=True,
        )

        # Add the documents to the index
        index.add_documents(
            documents_ids=metric_ids,
            documents_embeddings=documents_embeddings,
        )

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

        # Encode the query
        query_embeddings = self.model.encode(
            [query],
            batch_size=1,
            is_query=True,
            show_progress_bar=False,
        )

        # Retrieve results
        scores = self.retriever.retrieve(
            queries_embeddings=query_embeddings,
            k=k,
        )

        # Extract document IDs from the first query's results
        return [metric_name_to_class(hit["id"]) for hit in scores[0]]
