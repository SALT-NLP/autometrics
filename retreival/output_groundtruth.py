import csv

# Data representing the metric recommendations for each task and subtask.
data = {
    "SimpDA": {
        "fluency": [
            "UniEvalDialogue-understandability", "UniEvalDialogue-naturalness",
            "UniEvalSum-fluency", "BERTScoreP_roberta-large",
            "BERTScoreF_roberta-large", "UniEvalSum-relevance",
            "UniEvalSum-consistency", "BLEU", "UniEvalSum-coherence",
            "BERTScoreR_roberta-large"
        ],
        "meaning": [
            "BERTScoreF_roberta-large", "BERTScoreP_roberta-large",
            "BERTScoreR_roberta-large", "BARTScore_bart-large-cnn",
            "MOVERScore_distilbert-base-uncased", "CHRF",
            "LevenshteinRatio_max", "UniEvalSum-coherence",
            "ROUGE-L-p", "UniEvalSum-consistency"
        ],
        "simplicity": [
            "BERTScoreP_roberta-large", "UniEvalDialogue-understandability",
            "UniEvalDialogue-naturalness", "UniEvalSum-fluency",
            "UniEvalSum-relevance", "BERTScoreF_roberta-large",
            "BLEU", "UniEvalSum-consistency", "UniEvalSum-coherence",
            "ROUGE-1-r"
        ]
    },
    "SummEval": {
        "coherence": [
            "UniEvalSum-coherence", "UniEvalSum-relevance",
            "BERTScoreR_roberta-large", "BERTScoreF_roberta-large",
            "FKGL", "UniEvalSum-consistency", "ROUGE-L-p",
            "UniEvalSum-fluency", "BLEU", "CHRF"
        ],
        "consistency": [
            "UniEvalSum-consistency", "UniEvalSum-coherence",
            "UniEvalSum-relevance", "UniEvalFact-consistency",
            "ROUGE-L-p", "CHRF", "JaroSimilarity_max", "FKGL",
            "HammingDistance_min", "BERTScoreR_roberta-large"
        ],
        "fluency": [
            "UniEvalSum-coherence", "UniEvalSum-fluency",
            "BERTScoreR_roberta-large", "UniEvalSum-relevance",
            "UniEvalSum-consistency", "BERTScoreF_roberta-large",
            "LevenshteinRatio_max", "ROUGE-L-r", "ROUGE-2-f1",
            "BERTScoreP_roberta-large"
        ],
        "relevance": [
            "UniEvalSum-relevance", "UniEvalSum-coherence",
            "ROUGE-L-p", "BERTScoreR_roberta-large",
            "MOVERScore_distilbert-base-uncased", "BERTScoreF_roberta-large",
            "CHRF", "BLEU", "UniEvalSum-consistency",
            "LevenshteinRatio_max"
        ]
    }
}

# Write the data to a CSV file.
csv_filename = "groundtruth.csv"
with open(csv_filename, "w", newline="") as csvfile:
    fieldnames = ["task_name", "target", "best_metrics"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for task_name, targets in data.items():
        for target, metrics in targets.items():
            writer.writerow({
                "task_name": task_name,
                "target": target,
                "best_metrics": "['" + "','".join(metrics) + "']"
            })

# Read the CSV file and output its content.
with open(csv_filename, "r") as f:
    csv_content = f.read()

print(csv_content)