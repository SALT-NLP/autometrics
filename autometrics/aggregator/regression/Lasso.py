from sklearn.linear_model import Lasso as Las

from autometrics.aggregator.regression import Regression

class Lasso(Regression):
    def __init__(self, name=None, description=None, dataset=None, **kwargs):
        model = Las()

        if not name:
            name = "Lasso"

        if not description:
            description = "Lasso regression"

        super().__init__(name, description, model=model, dataset=dataset, **kwargs)
