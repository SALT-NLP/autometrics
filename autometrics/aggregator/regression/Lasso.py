from sklearn.linear_model import Lasso

from autometrics.aggregator.regression import Regression

class Lasso(Regression):
    def __init__(self, name=None, description=None):
        model = Lasso()

        if not name:
            name = "Lasso"

        if not description:
            description = "Lasso regression"

        super().__init__(name, description, model=model)
