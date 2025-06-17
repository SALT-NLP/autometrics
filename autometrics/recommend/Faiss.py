from typing import List, Type, Optional
import os
import json
import subprocess

from platformdirs import user_data_dir
from pyserini.search.faiss import FaissSearcher
from pyserini.encode import DocumentEncoder, QueryEncoder

from autometrics.metrics.Metric import Metric
from autometrics.dataset.Dataset import Dataset
from autometrics.recommend.MetricRecommender import MetricRecommender
from autometrics.recommend.utils import metric_name_to_class


class Faiss(MetricRecommender):
    """Metric recommender based on dense retrieval with a Faiss index built via Pyserini.

    This class mirrors the logic of `BM25` and `ColBERT`, but relies on Pyserini's
    dense encoders and a Faiss inner-product index.  The first instantiation will
    build the index (unless it already exists or *force_reindex* is *False*).  All
    subsequent instantiations reuse the persisted index, making start-up fast.
    """

    def __init__(
        self,
        metric_classes: List[Type[Metric]],
        index_path: str = user_data_dir("autometrics", "faiss"),
        encoder_name: str = "facebook/dpr-question_encoder-multiset-base",
        force_reindex: bool = False,
    ) -> None:
        self.metric_classes = metric_classes
        self.index_path = index_path
        self.encoder_name = encoder_name
        self.force_reindex = force_reindex

        # Build index if missing or forced.
        if force_reindex or not os.path.exists(self.index_path):
            self._build_index()

        # ------------------------------------------------------------------
        # Initialise the Pyserini Faiss searcher.
        # ------------------------------------------------------------------
        self.searcher = FaissSearcher(self.index_path)
        self.query_encoder = QueryEncoder(encoder_name=self.encoder_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_index(self) -> None:
        """Create a Faiss flat IP index for the metric docstrings."""
        print(
            f"[Faiss] Building dense index for {len(self.metric_classes)} metrics at {self.index_path} …"
        )

        # Ensure target directory exists (and is empty if we re-index).
        if os.path.exists(self.index_path):
            import shutil

            shutil.rmtree(self.index_path)
        os.makedirs(self.index_path, exist_ok=True)

        # ---------------------------------------------
        # 1) Write collection to JSONL (docs.jsonl)
        # ---------------------------------------------
        docs_file = os.path.join(self.index_path, "docs.jsonl")
        with open(docs_file, "w", encoding="utf-8") as f:
            for cls in self.metric_classes:
                doc = {
                    "id": cls.__name__,
                    "contents": cls.__doc__ or "",
                }
                json.dump(doc, f, ensure_ascii=False)
                f.write("\n")

        # ---------------------------------------------
        # 2) Encode and directly build Faiss flat index
        # ---------------------------------------------
        # We leverage Pyserini's CLI which handles batching, GPU utilisation, etc.
        # The `--to-faiss` flag converts vectors directly into a flat IP index.
        encode_cmd = [
            "python",
            "-m",
            "pyserini.encode",
            "input",
            "--corpus",
            docs_file,
            "--fields",
            "contents",
            "output",
            "--embeddings",
            self.index_path,
            "--to-faiss",
            "encoder",
            "--encoder",
            self.encoder_name,
            "--fields",
            "contents",
            "--batch",
            "32",
            "--fp16",
        ]

        result = subprocess.run(encode_cmd)
        if result.returncode != 0:
            raise RuntimeError("[Faiss] Failed to encode documents and build Faiss index.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def recommend(self, dataset: Dataset, target_measurement: str, k: int = 20) -> List[Type[Metric]]:
        task_desc = dataset.get_task_description()
        query = (
            f'I am looking for a metric to evaluate the following task: "{task_desc}" '
            f' In particular I care about "{target_measurement}".'
        )

        # Encode the query
        query_vector = self.query_encoder.encode(query)
        
        # Search using the encoded query
        hits = self.searcher.search(query_vector, k=k)
        return [metric_name_to_class(hit.docid) for hit in hits]
