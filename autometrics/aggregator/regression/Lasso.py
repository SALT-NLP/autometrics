from sklearn.linear_model import Lasso as Las

from autometrics.aggregator.regression import Regression

class Lasso(Regression):
    def __init__(self, name=None, description=None, dataset=None, **kwargs):
        # Use much lower alpha to avoid over-regularization
        # alpha=0.01 allows important features to be selected
        model = Las(alpha=0.01)

        if not name:
            name = "Lasso"

        if not description:
            description = "Lasso regression (alpha=0.01)"

        super().__init__(name, description, model=model, dataset=dataset, **kwargs)
