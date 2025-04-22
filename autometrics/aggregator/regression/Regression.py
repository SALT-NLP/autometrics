# Regression.py
import numpy as np
from autometrics.aggregator.Aggregator import Aggregator

class Regression(Aggregator):
    """
    Class for regression aggregation
    """
    def __init__(self, name, description, input_metrics=None, model=None, dataset=None, **kwargs):
        super().__init__(name, description, input_metrics, dataset, **kwargs)
        self.model = model

    def learn(self, dataset, target_column=None):
        """
        Learn the regression model
        """
        self.ensure_dependencies(dataset)
        df = dataset.get_dataframe()

        input_columns = self.get_input_columns()
        if not target_column:
            target_column = dataset.get_target_columns()[0]

        # Pull out X and y
        X = df[input_columns]
        y = df[target_column]

        # —— clip any +/-inf in X to the finite min/max of each column
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        mins = X_clean.min()
        maxs = X_clean.max()
        X = X.clip(lower=mins, upper=maxs, axis=1).fillna(0)

        # —— same for y (a Series)
        y_clean = y.replace([np.inf, -np.inf], np.nan)
        if y_clean.isna().all():
            # if everything was infinite, just zero out
            y = y.fillna(0)
        else:
            y = y.clip(lower=y_clean.min(), upper=y_clean.max()).fillna(0)

        # Now safe to fit
        self.model.fit(X, y)

    def _predict_unsafe(self, dataset, update_dataset=True):
        """
        Predict the target column
        """
        df = dataset.get_dataframe().copy()
        input_columns = self.get_input_columns()
        X = df[input_columns]

        y_pred = self.model.predict(X)

        if update_dataset:
            df.loc[:, self.name] = y_pred
            dataset.set_dataframe(df)

        return y_pred

    def identify_important_metrics(self):
        """
            Identify the most important metrics depending on the model.
            For linear models: Use coefficients.
            For tree-based models: Use feature importances.
        """
        metric_columns = self.get_input_columns()

        # Linear models (Ridge, Lasso, ElasticNet, PLS)
        if hasattr(self.model, 'coef_'):
            coef = self.model.coef_
            if coef.ndim == 1:
                pairs = zip(coef, metric_columns)
            else:
                pairs = zip(coef[0], metric_columns)
            return sorted(pairs, key=lambda x: abs(x[0]), reverse=True)

        # Tree-based models (RandomForest, GradientBoosting)
        if hasattr(self.model, 'feature_importances_'):
            pairs = zip(self.model.feature_importances_, metric_columns)
            return sorted(pairs, key=lambda x: abs(x[0]), reverse=True)

        raise ValueError(
            "The model does not support extracting feature importances or coefficients."
        )