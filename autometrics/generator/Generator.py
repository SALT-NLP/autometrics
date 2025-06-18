from abc import ABC, abstractmethod
import dspy
from typing import List
from autometrics.metrics.Metric import Metric

class Generator(ABC):

    def __init__(self, name, description, generator_llm: dspy.LM = None, executor_class: type = None, executor_kwargs: dict = None):
        self.name = name
        self.description = description
        self.generator_llm = generator_llm
        self.executor_class = executor_class
        self.executor_kwargs = executor_kwargs

    @abstractmethod
    def generate(self, dataset, target_measure: str, n_metrics: int = 5, **kwargs) -> List[Metric]:
        """
        Generate new metrics based on the dataset and task description
        """
        pass

    def get_name(self):
        return self.name
    
    def get_description(self):
        return self.description

    def __str__(self):
        return f"{self.name}: {self.description}"
    
    def __repr__(self):
        return f"{self.name}: {self.description}"