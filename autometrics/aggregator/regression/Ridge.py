from sklearn.linear_model import Ridge as Rid

from autometrics.aggregator.regression import Regression

class Ridge(Regression):
    def __init__(self, name=None, description=None, dataset=None, **kwargs):
        model = Rid()

        if not name:
            name = "Ridge"

        if not description:
            description = "Ridge regression"

        super().__init__(name, description, model=model, dataset=dataset, **kwargs)
