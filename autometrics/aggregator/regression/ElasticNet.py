from sklearn.linear_model import ElasticNet

from autometrics.aggregator.regression import Regression

class ElasticNet(Regression):
    def __init__(self, name=None, description=None):
        model = ElasticNet()

        if not name:
            name = "ElasticNet"

        if not description:
            description = "ElasticNet regression"

        super().__init__(name, description, model=model)
