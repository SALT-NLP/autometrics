from sklearn.ensemble import RandomForestRegressor

from autometrics.aggregator.regression import Regression

class RandomForest(Regression):
    def __init__(self, name=None, description=None):
        model = RandomForestRegressor()

        if not name:
            name = "RandomForest"

        if not description:
            description = "RandomForest regression"

        super().__init__(name, description, model=model)
