from autometrics.dataset import Dataset
from scipy.stats import spearmanr

def calculate_correlation(dataset: Dataset, group_by: str = None, correlation=spearmanr, compute_all=False) -> dict:
    '''
    Calculate the correlation for the given dataset.

    Parameters:
    -----------
    dataset : Dataset
        The dataset object containing the data to be analyzed. It should have methods to get the dataframe, target columns, ignore columns, and metric columns.
    group_by : str, optional
        The column name by which to group the data before calculating the Spearman correlation. Default is 'None'.
    correlation : function, optional
        The correlation function to use. Default is `spearmanr` from `scipy.stats`.

    Returns:
    --------
    dict
        A dictionary containing the Spearman correlation coefficients. If `group_by` is specified, the dictionary will contain the average Spearman correlation for each target column and metric column, grouped by the specified column. If `group_by` is not specified, the dictionary will contain the Spearman correlation for each target column and metric column for the entire dataset.
        Dictionary structure:
            {
                target_column_1: {
                    metric_column_1: correlation_1,
                    metric_column_2: correlation_2,
                    ...
                },
                target_column_2: {
                    metric_column_1: correlation_1,
                    metric_column_2: correlation_2,
                    ...
                },
                ...
            }
    '''
    df = dataset.get_dataframe()
    target_columns = dataset.get_target_columns()
    ignore_columns = dataset.get_ignore_columns()
    if compute_all:
        metric_columns = [col for col in df.columns if col not in ignore_columns and col not in target_columns]
    else:
        metric_columns = dataset.get_metric_columns()

    if metric_columns is None:
        metric_columns = [col for col in df.columns if col not in ignore_columns and col not in target_columns]

    if group_by:
        # group by the column 'group_by' and calculate the average spearman correlation for each target_column and metric_column.
        ignore_columns_minus_group_id = [col for col in ignore_columns if col != group_by]
        grouped_df = df.drop(columns=ignore_columns_minus_group_id).groupby([group_by])
        spearman_correlations_grouped = {}
        for target_column in target_columns:
            spearman_correlations_grouped[target_column] = {}
            for metric_column in metric_columns:
                spearman_correlations_grouped[target_column][metric_column] = grouped_df.apply(lambda x: correlation(x[target_column], x[metric_column])[0]).mean()

        return spearman_correlations_grouped

    else:
        # for each of the target_columns, calculate the spearman correlation with each of the metric_columns.
        spearman_correlations_all = {}
        for target_column in target_columns:
            spearman_correlations_all[target_column] = {}
            for metric_column in metric_columns:
                spearman_correlations_all[target_column][metric_column] = correlation(df[target_column], df[metric_column])[0]

        return spearman_correlations_all
