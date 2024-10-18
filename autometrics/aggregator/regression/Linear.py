from sklearn.linear_model import LinearRegression

from autometrics.aggregator.regression import Regression

class Linear(Regression):
    def __init__(self, name=None, description=None):
        model = LinearRegression()

        if not name:
            name = "LinearRegression"

        if not description:
            description = "Linear regression"

        super().__init__(name, description, model=model)
