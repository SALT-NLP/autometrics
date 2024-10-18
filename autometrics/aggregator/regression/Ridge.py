from sklearn.linear_model import Ridge

from autometrics.aggregator.regression import Regression

class Ridge(Regression):
    def __init__(self, name=None, description=None):
        model = Ridge()

        if not name:
            name = "Ridge"

        if not description:
            description = "Ridge regression"

        super().__init__(name, description, model=model)
