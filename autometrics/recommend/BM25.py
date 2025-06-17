from typing import List, Type
from autometrics.metrics.Metric import Metric
from autometrics.dataset.Dataset import Dataset
from autometrics.recommend.MetricRecommender import MetricRecommender
import os
from platformdirs import user_data_dir
import json
import subprocess
from pyserini.search.lucene import LuceneSearcher
from autometrics.recommend.utils import metric_name_to_class

class BM25(MetricRecommender):
    def __init__(self, metric_classes: List[Type[Metric]], index_path: str = user_data_dir("autometrics", "bm25"), force_reindex: bool = False):
        self.metric_classes = metric_classes
        self.index_path = index_path
        self.force_reindex = force_reindex

        if not os.path.exists(index_path) or force_reindex:
            print(f"Index not found at {index_path} (or force_reindex is True). Reindexing {len(metric_classes)} metrics.")

            if not os.path.exists(index_path):
                os.makedirs(index_path)

            metric_names = [metric.__name__ for metric in metric_classes]
            metric_docs = [metric.__doc__ for metric in metric_classes]

            with open(os.path.join(index_path, "docs.jsonl"), "w") as f:
                for metric_name, metric_doc in zip(metric_names, metric_docs):
                    json.dump({"id": metric_name, "contents": metric_doc}, f)
                    f.write("\n")

            result = subprocess.run([
                "python", "-m", "pyserini.index.lucene",
                "--collection", "JsonCollection",
                "--input", index_path,
                "--index", "bm25",
                "--generator", "DefaultLuceneDocumentGenerator", 
                "--threads", "1",
                "--storePositions", "--storeDocvectors", "--storeRaw"
            ], check=True)

            if result.returncode != 0:
                raise RuntimeError("Failed to build BM25 index")

        self.searcher = LuceneSearcher(index_path)
        self.searcher.set_language('en')

    def recommend(self, dataset: Dataset, target_measurement: str, k: int = 20) -> List[Type[Metric]]:
        query = f'I am looking for a metric to evaluate the following task: "{dataset.get_task_description()}"  In particular I care about "{target_measurement}".'
        hits = self.searcher.search(query, k=k)
        return [metric_name_to_class(hit.docid) for hit in hits]