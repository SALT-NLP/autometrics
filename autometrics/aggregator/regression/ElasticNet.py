from sklearn.linear_model import ElasticNet as ENet

from autometrics.aggregator.regression import Regression

class ElasticNet(Regression):
    def __init__(self, name=None, description=None, dataset=None, **kwargs):
        # Use much lower alpha to avoid over-regularization
        # alpha=0.01 allows important features to be selected
        # l1_ratio=0.5 gives equal weight to L1 and L2 regularization
        model = ENet(alpha=0.01, l1_ratio=0.5)

        if not name:
            name = "ElasticNet"

        if not description:
            description = "ElasticNet regression (alpha=0.01, l1_ratio=0.5)"

        super().__init__(name, description, model=model, dataset=dataset, **kwargs)
