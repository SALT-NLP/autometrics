from autometrics.aggregator.regression import Regression

class BudgetRegression(Regression):
    """
    Class for regression aggregation
    """
    def __init__(self, regressor: Regression, metric_budget: int, name=None, description=None):
        """
        Initialize the class
        """
        self.regressor = regressor
        self.model = regressor.model
        self.name = regressor.name + f"Top {metric_budget}" if not name else name
        self.description = regressor.description + f"Top {metric_budget}" if not description else description

        super().__init__(self.name, self.description, model=self.model)

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

        important_metrics = self.identify_important_metrics()[:self.metric_budget]

        self.input_metrics = [metric for metric in self.input_metrics if metric.get_name() in important_metrics]

        self.model.fit(X[important_metrics], y)
