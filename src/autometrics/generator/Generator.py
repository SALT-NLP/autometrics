from abc import ABC, abstractmethod

class Generator(ABC):

    def __init__(self, name, description):
        self.name = name
        self.description = description

    @abstractmethod
    def generate(self, dataset, **kwargs):
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
        return