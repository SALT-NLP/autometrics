# %%
from autometrics.dataset.datasets.simplification import SimpDA
from autometrics.dataset.datasets.summeval import summeval
from autometrics.dataset.datasets.primock57 import primock57
from autometrics.dataset.datasets.helpsteer import helpsteer

from autometrics.util.analysis import display_top_5_metrics_by_validation, get_top_metric_by_validation, plot_metric_target_scatterplot
from autometrics.evaluate.correlation import calculate_correlation
from autometrics.metrics.MetricBank import all_metrics, reference_free_metrics

from scipy.stats import pearsonr

# %%
def get_top_metrics(dataset, bank=all_metrics):
    """
    Get the top metric for each validation set in the dataset.
    """
    calculate_correlation(dataset, correlation=pearsonr)
    train, dev, test = dataset.get_splits(train_ratio=0.2, val_ratio=0.3, seed=42)

    dev.add_metrics(bank)
    test.add_metrics(bank)
    top_metrics = get_top_metric_by_validation(dev, test, False)
    display_top_5_metrics_by_validation(dev, test, False)
    print("Top metrics for each validation set:")
    print(top_metrics)
    return top_metrics
    

# %%
print("SimpDA")
top_simpda = get_top_metrics(SimpDA())

# %%
print("SummEval")
top_summeval = get_top_metrics(summeval())

# %%
print("Primock57")
top_primock57 = get_top_metrics(primock57())

# %%
print("HelpSteer")
top_helpsteer = get_top_metrics(helpsteer(), bank=reference_free_metrics)
