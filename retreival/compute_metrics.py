import pandas as pd
import ast
import math
import csv

DEBUG = True

# Mapping dictionary: keys are ground truth prefixes; values are the recommended file names.
metric_mapping = {
    "FKGL": "FKGL.md",
    "UniEvalSum": "UniEvalSum.md",
    "SARI": "SARI.md",
    "Perplexity": "Perplexity.md",
    "BARTScore": "BARTScore.md",
    "BERTScore": "BERTScore.md",
    "MOVERScore": "MoverScore.md",
    "MoverScore": "MoverScore.md",  # variant capitalization
    "ROUGE": "ROUGE.md",
    "CHRF": "CHRF.md",
    "BLEU": "BLEU.md",
    "GLEU": "GLEU.md",
    "CIDEr": "CIDEr.md",
    "TER": "TER.md",
    "UniEvalDialogue": "UniEvalDialogue.md",
    "UniEvalFact": "UniEvalFact.md",
    "LevenshteinRatio": "LevenshteinRatio.md",
    "LevenshteinDistance": "LevenshteinDistance.md",
    "HammingDistance": "HammingDistance.md",
    "JaroWinklerSimilarity": "JaroWinklerSimilarity.md",
    "JaroSimilarity": "JaroSimilarity.md"
}

def clean_metric(metric_name):
    """Remove extraneous brackets, quotes, and whitespace."""
    return metric_name.strip("[]'\" ")

def map_metric(metric_name):
    """Map a ground truth metric name (after cleaning) to a recommended file name using prefix matching."""
    clean = clean_metric(metric_name)
    for prefix, rec in metric_mapping.items():
        if clean.startswith(prefix):
            if DEBUG:
                print(f"Mapping '{clean}' matched prefix '{prefix}' -> '{rec}'")
            return rec
    if DEBUG:
        print(f"Mapping '{clean}' found no match.")
    return None

def unique_list(lst):
    """Return the list with duplicates removed, preserving order."""
    seen = set()
    unique = []
    for item in lst:
        if item not in seen:
            unique.append(item)
            seen.add(item)
    return unique

def unique_predictions(predicted):
    """Deduplicate the predicted list while preserving order."""
    return unique_list(predicted)

def compute_recall(predicted, relevant_list, k):
    """
    Compute recall@k: (# unique relevant items in top k) / (# unique ground truth relevant items).
    Both predicted and relevant_list are assumed to be deduplicated.
    """
    unique_pred = unique_predictions(predicted)
    top_k = unique_pred[:k]
    count = sum(1 for metric in top_k if metric in relevant_list)
    recall_value = count / len(relevant_list) if relevant_list else 0
    if DEBUG:
        print(f"Recall@{k}: top {k} predictions (deduplicated) = {top_k}")
        print(f"Relevant list: {relevant_list}")
        print(f"Count of relevant items in top {k}: {count} -> recall: {recall_value}")
    return recall_value

def compute_mrr(predicted, relevant_list):
    """
    Compute Mean Reciprocal Rank (MRR) for the deduplicated predicted list.
    Returns the reciprocal rank of the first relevant item.
    """
    unique_pred = unique_predictions(predicted)
    for idx, metric in enumerate(unique_pred, start=1):
        if metric in relevant_list:
            if DEBUG:
                print(f"MRR: Found relevant metric '{metric}' at rank {idx} -> MRR: {1/idx}")
            return 1 / idx
    if DEBUG:
        print("MRR: No relevant metric found in predictions.")
    return 0

def compute_ndcg(predicted, relevant_list, k):
    """
    Compute NDCG@k for the deduplicated predicted list using binary relevance.
    """
    unique_pred = unique_predictions(predicted)
    dcg = 0.0
    for i, metric in enumerate(unique_pred[:k], start=1):
        rel = 1 if metric in relevant_list else 0
        dcg += rel / math.log2(i + 1)
    ideal_count = min(len(relevant_list), k)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, ideal_count + 1))
    ndcg_value = dcg / idcg if idcg > 0 else 0
    if DEBUG:
        print(f"NDCG@{k}: DCG = {dcg}, IDCG = {idcg}, NDCG = {ndcg_value}")
    return ndcg_value

def parse_gt_list(x):
    """
    Parse the ground truth string into a list.
    If it's a nested list (e.g., [['metric1', 'metric2', ...]]),
    return the inner list.
    """
    try:
        parsed = ast.literal_eval(x)
        if isinstance(parsed, list):
            if parsed and isinstance(parsed[0], list):
                if DEBUG:
                    print(f"Parsed nested list: {parsed} -> using inner list {parsed[0]}")
                return parsed[0]
            return parsed
        else:
            if DEBUG:
                print(f"Parsed value is not a list: {parsed}")
            return []
    except Exception as e:
        if DEBUG:
            print(f"Error parsing gt_list: {x} - {e}")
        return []

# -----------------------------
# Load ground truth CSV file.
# -----------------------------
if DEBUG:
    print("Loading ground truth CSV file: 'groundtruth.csv'")
gt_df = pd.read_csv("groundtruth.csv")
# Parse the 'best_metrics' column into a list.
gt_df['gt_list'] = gt_df['best_metrics'].apply(parse_gt_list)
if DEBUG:
    print("Ground truth metrics (parsed):")
    print(gt_df[['task_name', 'target', 'gt_list']])

# Map ground truth metrics and then take the first 5 (unique) as the relevant items.
gt_df['mapped_gt'] = gt_df['gt_list'].apply(lambda lst: [map_metric(metric) for metric in lst])
if DEBUG:
    print("Mapped ground truth metrics:")
    print(gt_df[['task_name', 'target', 'mapped_gt']])
# Create a unique list from the top 5 mapped metrics.
gt_df['relevant_list'] = gt_df['mapped_gt'].apply(lambda lst: unique_list(lst[:5]))
if DEBUG:
    print("Unique relevant list (top 5) for each row:")
    print(gt_df[['task_name', 'target', 'relevant_list']])

# -----------------------------
# Define the systems and corresponding CSV files.
# -----------------------------
systems = {
    "bm25": "predictions_bm25.csv",
    "colbert": "predictions_colbert.csv",
    "llm": "predictions_llm.csv"
}

# List to store evaluation results.
results = []

# Loop over each system.
for system, filename in systems.items():
    if DEBUG:
        print(f"\nProcessing system: {system}, file: {filename}")
    # Load predictions CSV.
    pred_df = pd.read_csv(filename)
    # 'recommendations' is a string representation of a list.
    pred_df['pred_list'] = pred_df['recommendations'].apply(lambda x: ast.literal_eval(x))
    if DEBUG:
        print("Prediction data sample:")
        print(pred_df.head())
    
    # Join on task_name and target.
    merged = pd.merge(gt_df, pred_df, on=['task_name', 'target'], how='inner')
    if DEBUG:
        print(f"Merged data for system '{system}':")
        print(merged[['task_name', 'target', 'mapped_gt', 'pred_list']])
    
    # Compute metrics for each task-target pair.
    for idx, row in merged.iterrows():
        if DEBUG:
            print(f"\nEvaluating {system} - {row['task_name']} / {row['target']}")
        pred_list = row['pred_list']
        relevant_list = row['relevant_list']
        if DEBUG:
            print(f"Predicted list (raw): {pred_list}")
            print(f"Unique predicted list: {unique_predictions(pred_list)}")
            print(f"Relevant list: {relevant_list}")
        
        recall1  = compute_recall(pred_list, relevant_list, 1)
        recall5  = compute_recall(pred_list, relevant_list, 5)
        recall10 = compute_recall(pred_list, relevant_list, 10)
        recall15 = compute_recall(pred_list, relevant_list, 15)
        
        mrr    = compute_mrr(pred_list, relevant_list)
        ndcg15 = compute_ndcg(pred_list, relevant_list, 15)
        
        results.append({
            "system": system,
            "task_name": row["task_name"],
            "target": row["target"],
            "recall@1": recall1,
            "recall@5": recall5,
            "recall@10": recall10,
            "recall@15": recall15,
            "MRR": mrr,
            "NDCG@15": ndcg15
        })

# Convert the results to a DataFrame and output to CSV.
results_df = pd.DataFrame(results)
results_df.to_csv("evaluation_results.csv", index=False)
print("\nEvaluation results saved to evaluation_results.csv")