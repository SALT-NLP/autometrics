from abc import ABC, abstractmethod
import json
from autometrics.metrics.reference_based.ReferenceBasedMetric import ReferenceBasedMetric
    
class GeneratedRefBasedMetric(ReferenceBasedMetric):
    def __init__(self, name, description, metric_card=None, **kwargs):
        # Pass all parameters explicitly to parent constructor for caching
        if metric_card is None:
            self.metric_card = self._generate_metric_card()
        else:
            self.metric_card = metric_card

        self.kwargs = kwargs

        super().__init__(name, description, **kwargs)

    @abstractmethod
    def _calculate_impl(self, input, output, references=None, **kwargs):
        pass

    def _calculate_batched_impl(self, inputs, outputs, references=None, **kwargs):
        return [self._calculate_impl(input, output, references, **kwargs) for input, output in zip(inputs, outputs)]

    @abstractmethod
    def _generate_metric_card(self):
        pass

    @abstractmethod
    def save_python_code(self, path: str):
        pass

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
        return f"GeneratedRefBasedMetric(name={self.name}, description={self.description})\n{self.metric_card}"
