from sklearn.linear_model import Ridge as Rid

from autometrics.aggregator.regression import Regression

class Ridge(Regression):
    def __init__(self, name=None, description=None, dataset=None, **kwargs):
        # Use much lower alpha to avoid over-regularization
        # alpha=0.01 allows important features to be selected
        model = Rid(alpha=0.01)

        if not name:
            name = "Ridge"

        if not description:
            description = "Ridge regression (alpha=0.01)"

        super().__init__(name, description, model=model, dataset=dataset, **kwargs)
