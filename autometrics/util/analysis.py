from autometrics.evaluate.correlation import calculate_correlation
import pandas as pd

def abbreviate_metric_name(metric_name):
    # Define abbreviations for common long parts of the names
    return metric_name.replace("inc_plus_omi_", "ipo_").replace("predictions_", "pred_").replace("ElasticNet", "ENet").replace("GradientBoosting", "GB").replace("Ridge", "Rg").replace("Lasso", "L").replace("PLS", "PLS")

def top_5_metrics_by_validation(validation_dataset, test_dataset, compute_all=False):
    """
    Returns the top 5 metrics by validation score

    Parameters:
    -----------
    validation_dataset : Dataset
        The dataset object containing the validation data.
    test_dataset : Dataset
        The dataset object containing the test data.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the top 5 metrics by validation score for each target category, along with the test score.
    """

    top_correlations = {}

    validation_data = calculate_correlation(validation_dataset, compute_all=compute_all)
    test_data = calculate_correlation(test_dataset, compute_all=compute_all)
    
    # Iterate over each target category (time_sec, inc_plus_omi, etc.)
    for target_column, val_data in validation_data.items():
        # Sort validation correlations and get top 5
        sorted_val_correlations = sorted(val_data.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        
        # Prepare row for this target
        top_correlations[target_column] = []
        
        for metric, val_corr in sorted_val_correlations:
            test_corr = test_data.get(target_column, {}).get(metric, "N/A")
            metric_abbr = abbreviate_metric_name(metric)
            # Format as (metric_abbr, val_corr, test_corr)
            top_correlations[target_column].append(f"{metric_abbr} ({test_corr})")

    # Create a DataFrame from the dictionary
    df = pd.DataFrame.from_dict(top_correlations, orient='index', columns=[f'Top {i+1} Metric & Value' for i in range(5)])
    
    return df