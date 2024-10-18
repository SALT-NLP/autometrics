from sklearn.linear_model import ElasticNet as ENet

from autometrics.aggregator.regression import Regression

class ElasticNet(Regression):
    def __init__(self, name=None, description=None, dataset=None, **kwargs):
        model = ENet()

        if not name:
            name = "ElasticNet"

        if not description:
            description = "ElasticNet regression"

        super().__init__(name, description, model=model, dataset=dataset, **kwargs)
