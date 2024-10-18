from sklearn.cross_decomposition import PLSRegression

from autometrics.aggregator.regression import Regression

class PLS(Regression):
    def __init__(self, n_components=2, name=None, description=None):
        self.n_components = n_components
        model = PLSRegression(n_components=n_components)

        if not name:
            name = f"PLS_{n_components}"

        if not description:
            description = f"PLS with {n_components} components"

        super().__init__(name, description, model=model)