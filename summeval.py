# %%
from autometrics.dataset.datasets.summeval.summeval import SummEval
from autometrics.util.analysis import display_top_5_metrics_by_validation, get_top_metric_by_validation, plot_metric_target_scatterplot
from autometrics.evaluate.correlation import calculate_correlation
from autometrics.aggregator.regression.PLS import PLS
from autometrics.aggregator.regression import Ridge, ElasticNet, Lasso, RandomForest, GradientBoosting, Linear
from autometrics.aggregator.regression.BudgetRegression import BudgetRegression
from autometrics.metrics.MetricBank import all_metrics

# %%
dataset = SummEval()
dataset.add_metrics(all_metrics)

# %%
from scipy.stats import pearsonr

calculate_correlation(dataset, correlation=pearsonr)

# %%
train, dev, test = dataset.get_splits(train_ratio=0.2, val_ratio=0.3, seed=42)

# %%
display_top_5_metrics_by_validation(dev, test, False).to_csv("summ_eval_starting_metrics.csv", index=True)

# %%
for target_column in dataset.target_columns:
    metric = get_top_metric_by_validation(dev, target_column, False)
    plot_metric_target_scatterplot(test, metric, target_column)

# %%
for model in [PLS, ElasticNet, Lasso, Ridge, RandomForest, GradientBoosting, Linear]:
    for target_column in dataset.target_columns:
        model_instance = model(dataset=train, name=model.__name__ + '_' + target_column)
        model_instance.learn(train, target_column)
        model_instance.predict(train, target_column)
        model_instance.predict(dev, target_column)
        model_instance.predict(test, target_column)

# %%
display_top_5_metrics_by_validation(dev, test, True).to_csv("summ_eval_dev_regression.csv", index=True)

# %%
display_top_5_metrics_by_validation(test, test, True).to_csv("summ_eval_test_regression.csv", index=True)

# %%
for target_column in dataset.target_columns:
    metric = get_top_metric_by_validation(dev, target_column, True)
    plot_metric_target_scatterplot(test, metric, target_column)


# %%
# Budget Runs
for model in [PLS, ElasticNet, Lasso, Ridge, RandomForest, GradientBoosting, Linear]:
    for budget in [2, 3, 5, 10, 15, 20, 25]:
        for target_column in dataset.target_columns:
            model_instance = model(dataset=train, name=model.__name__ + '_' + target_column)
            budget_model = BudgetRegression(model_instance, budget)
            budget_model.learn(train, target_column)
            budget_model.predict(train, target_column)
            budget_model.predict(dev, target_column)
            budget_model.predict(test, target_column)

# %%
display_top_5_metrics_by_validation(dev, test, True).to_csv("summ_eval_dev_budget.csv", index=True)

# %%
display_top_5_metrics_by_validation(test, test, True).to_csv("summ_eval_test_budget.csv", index=True)

# %%
for target_column in dataset.target_columns:
    metric = get_top_metric_by_validation(dev, target_column, True)
    plot_metric_target_scatterplot(test, metric, target_column)


