from abc import ABC, abstractmethod
import json
from autometrics.metrics.reference_free.ReferenceFreeMetric import ReferenceFreeMetric
import dspy
import re

class GeneratedRefFreeMetric(ReferenceFreeMetric):
    def __init__(self, name, description, metric_card=None, metric_card_author_model: dspy.LM = None, **kwargs):

        self.name = name
        self.description = description
        self.metric_card_author_model = metric_card_author_model

        # Pass all parameters explicitly to parent constructor for caching
        if metric_card is None:
            self.metric_card = self._generate_metric_card(metric_card_author_model)
            self.__doc__ = self.metric_card
        else:
            self.metric_card = metric_card

        self.kwargs = kwargs

        super().__init__(name, description, **kwargs)

        self.exclude_from_cache_key("metric_card_author_model", "metric_card")

    @abstractmethod
    def _calculate_impl(self, input, output, references=None, **kwargs):
        pass

    def _calculate_batched_impl(self, inputs, outputs, references=None, **kwargs):
        return [self._calculate_impl(input, output, references, **kwargs) for input, output in zip(inputs, outputs)]

    @abstractmethod
    def _generate_metric_card(self, author_model: dspy.LM = None):
        pass

    @abstractmethod
    def _generate_python_code(self, include_metric_card: bool = True) -> str:
        pass

    def save_python_code(self, path: str) -> str:
        code = self._generate_python_code()
        # Only sanitize the filename portion, not the full path
        base_path = str(path)
        filename = base_path.replace('\\', '/').split('/')[-1]
        safe_filename = re.sub(r'[^\w\-_\.]', '_', filename)
        safe_path = base_path[:-len(filename)] + safe_filename

        with open(safe_path, "w", encoding="utf-8") as f:
            f.write(code)

        return safe_path

    def save(self, path: str):
        output = {
            "name": self.name, 
            "description": self.description, 
            "metric_card": self.metric_card, 
            **self.kwargs
        }
        with open(path, "w") as f:
            json.dump(output, f, indent=4)

    @classmethod
    def load(cls, path: str):
        with open(path, "r") as f:
            output = json.load(f)
        return cls(
            name=output["name"],
            description=output["description"],
            metric_card=output["metric_card"],
            **output["kwargs"]
        )

    def __repr__(self):
        return f"GeneratedRefFreeMetric(name={self.name}, description={self.description})\n{self.metric_card}"
