from typing import List, Type
from autometrics.metrics.Metric import Metric
from autometrics.dataset.Dataset import Dataset
from autometrics.recommend.MetricRecommender import MetricRecommender
import os
from platformdirs import user_data_dir
import json
import subprocess
from pyserini.search.lucene import LuceneSearcher

class BM25(MetricRecommender):
    def __init__(self, metric_classes: List[Type[Metric]], index_path: str = user_data_dir("autometrics", "bm25"), force_reindex: bool = False):
        self.metric_classes = metric_classes
        # Root directory that will hold both collection and index
        self.root_path = index_path
        self.force_reindex = force_reindex

        # ------------------------------------------------------------------
        # Directory layout
        #   <root_path>/collection/docs.jsonl   → input to indexer
        #   <root_path>/index/                 → Lucene index written here
        # ------------------------------------------------------------------
        self.collection_path = os.path.join(self.root_path, "collection")
        self.lucene_index_path = os.path.join(self.root_path, "index")

        # (Re-)build index if needed
        if force_reindex or not os.path.exists(self.lucene_index_path):
            print(
                f"Building BM25 index in {self.lucene_index_path} for {len(metric_classes)} metrics …"
            )

            # Clean slate
            import shutil
            if os.path.exists(self.root_path):
                shutil.rmtree(self.root_path)
            os.makedirs(self.collection_path, exist_ok=True)

            # Write docs.jsonl
            metric_names = [m.__name__ for m in metric_classes]
            metric_docs = [m.__doc__ or "" for m in metric_classes]

            docs_file = os.path.join(self.collection_path, "docs.jsonl")
            with open(docs_file, "w", encoding="utf-8") as f:
                for name, doc in zip(metric_names, metric_docs):
                    json.dump({"id": name, "contents": doc}, f)
                    f.write("\n")

            # Invoke Pyserini indexer
            result = subprocess.run([
                "python", "-m", "pyserini.index.lucene",
                "--collection", "JsonCollection",
                "--input", self.collection_path,
                "--index", self.lucene_index_path,
                "--generator", "DefaultLuceneDocumentGenerator",
                "--threads", "1",
                "--storePositions", "--storeDocvectors", "--storeRaw",
            ])

            if result.returncode != 0:
                raise RuntimeError("Failed to build BM25 index")

        # ------------------------------------------------------------------
        # Initialise Lucene searcher on the freshly built (or cached) index.
        # ------------------------------------------------------------------
        self.searcher = LuceneSearcher(self.lucene_index_path)
        self.searcher.set_language('en')

        super().__init__(metric_classes, index_path, force_reindex)

    def recommend(self, dataset: Dataset, target_measurement: str, k: int = 20) -> List[Type[Metric]]:
        query = f'I am looking for a metric to evaluate the following task: "{dataset.get_task_description()}"  In particular I care about "{target_measurement}".'
        hits = self.searcher.search(query, k=k)
        return [self.metric_name_to_class(hit.docid) for hit in hits]