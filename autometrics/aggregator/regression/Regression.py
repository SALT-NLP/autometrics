from autometrics.aggregator.Aggregator import Aggregator

class Regression(Aggregator):
    """
    Class for regression aggregation
    """
    def __init__(self, name, description, input_metrics=None, model=None):
        super().__init__(name, description, input_metrics)
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

        X = df[input_columns]
        y = df[target_column]

        self.model.fit(X, y)

    def _predict_unsafe(self, dataset, update_dataset=True):
        """
        Predict the target column
        """

        df = dataset.get_dataframe()

        input_columns = self.get_input_columns()
        X = df[input_columns]

        y = self.model.predict(X)

        if update_dataset:
            df[self.name] = y
            dataset.set_dataframe(df)

        return y
    
    def identify_important_metrics(self):
        '''
            Identify the most important metrics depending on the model.
            For linear models: Use coefficients.
            For tree-based models: Use feature importances.
        '''
        metric_columns = self.get_input_columns()

        # Linear models (Ridge, Lasso, ElasticNet, PLS)
        if hasattr(self.model, 'coef_'):
            coef = self.model.coef_

            # Check if coef is 1D or 2D and handle accordingly
            if len(coef.shape) == 1:
                return sorted(zip(coef, metric_columns), key=lambda x: abs(x[0]), reverse=True)
            else:
                return sorted(zip(coef[0], metric_columns), key=lambda x: abs(x[0]), reverse=True)
        
        # Tree-based models (RandomForest, GradientBoosting)
        elif hasattr(self.model, 'feature_importances_'):
            return sorted(zip(self.model.feature_importances_, metric_columns), key=lambda x: abs(x[0]), reverse=True)

        else:
            raise ValueError("The model does not support extracting feature importances or coefficients.")