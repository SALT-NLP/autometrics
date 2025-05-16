from autometrics.dataset.datasets.simplification import SimpDA
from autometrics.dataset.datasets.summeval.summeval import SummEval
from autometrics.dataset.datasets.primock57.primock57 import Primock57
from autometrics.dataset.datasets.helpsteer.helpsteer import HelpSteer

import pandas as pd
from ragatouille import RAGPretrainedModel

RAG = RAGPretrainedModel.from_index("retreival/.ragatouille/colbert/indexes/all_metrics")

tasks = [
    SimpDA(),
    SummEval(),
    Primock57(),
    HelpSteer()
]
task_descriptions = {
    "SimpDA": "Given a complicated original sentence, simplify it in a way such that a broader audience could easily understand it.",
    "SummEval": "Summarize the given text in a way that captures the main ideas and themes.",
    "primock57": "Write a high quality clinical note based on the transcript of a consultation with a patient.",
    "HelpSteer": "Answer the user query as a helpful chatbot assistant."
}

def get_top_k_hits(task_name, target, k=20):
    """
    Retrieve the top k hits for a given task name using the BM25 searcher.
    
    :param task_name: The name of the task to retrieve hits for.
    :param k: The number of top hits to retrieve.
    :return: A list of top k hits.
    """
    if task_name not in task_descriptions:
        raise ValueError(f"Task '{task_name}' is not recognized.")
    
    query = f'I am looking for a metric to evaluate the following task: "{task_descriptions[task_name]}"  In particular I care about "{target}".'
    results = RAG.search(query, k=k)
    
    return results

output_file = 'retreival/predictions_colbert.csv'
output_dict = {"task_name": [], "target": [], "recommendations": []}

for task in tasks:
    task_name = task.name
    targets = task.get_target_columns()

    for target in targets:
        print(f"Task: {task_name}, Target: {target}")
        top_hits = get_top_k_hits(task_name, target, k=20)
        doc_ids = [hit["document_id"] for hit in top_hits]
        print(f"Top 20 hits for target '{target}': {doc_ids}")
        print("-" * 80)
        print("\n")

        output_dict["task_name"].append(task_name)
        output_dict["target"].append(target)
        output_dict["recommendations"].append(doc_ids)

# Save the results to a CSV file
output_df = pd.DataFrame(output_dict)
output_df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")