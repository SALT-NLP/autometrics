# %%
from autometrics.dataset.datasets.simplification import SimpDA
from autometrics.dataset.datasets.summeval.summeval import SummEval
from autometrics.dataset.datasets.primock57.primock57 import Primock57
from autometrics.dataset.datasets.helpsteer.helpsteer import HelpSteer

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
    targets = test.get_target_columns()
    mapping = {}
    for target in targets:
        mapping[target] = get_top_metric_by_validation(dev, target, False, 10)

    display_top_5_metrics_by_validation(dev, test, False)
    print("Top metrics for each validation set:")
    print(mapping)

    return mapping

# %%
# print("SimpDA")
# top_simpda = get_top_metrics(SimpDA())

# # %%
# print("SummEval")
# top_summeval = get_top_metrics(SummEval())

# %%
print("Primock57")
top_primock57 = get_top_metrics(Primock57())

# %%
print("HelpSteer")
top_helpsteer = get_top_metrics(HelpSteer(), bank=reference_free_metrics)
