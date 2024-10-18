from sklearn.cross_decomposition import PLSRegression

from autometrics.aggregator.regression import Regression

class PLS(Regression):
    def __init__(self, name=None, description=None, n_components=2, dataset=None, **kwargs):
        model = PLSRegression(n_components=n_components)

        if not name:
            name = f"PLS_{n_components}"

        if not description:
            description = f"PLS with {n_components} components"

        # Pass name, description, and model to the parent class using keyword argument for model
        super().__init__(name, description, model=model, dataset=dataset, **kwargs)
        self.n_components = n_components