from create_colbert_index import get_docs
import os

from autometrics.dataset.datasets.simplification import SimpDA
from autometrics.dataset.datasets.summeval.summeval import SummEval
from autometrics.dataset.datasets.primock57.primock57 import Primock57
from autometrics.dataset.datasets.helpsteer.helpsteer import HelpSteer

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

doc_ids, documents = get_docs("autometrics/metrics/documentation/")

output_folder = "retreival/prompts/"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

PROMPT_TEMPLATE = """ \
I am looking for a metric to evaluate the following task: "{task_description}".
In particular I care about "{target}".

Here are some metrics that might be relevant:
{documents}

Please rank the metrics from most relevant to least relevant for the task and target above.
You can reason first about what makes a metric relevant for the task and target, and then provide your ranking.

The final ranking should just be a python list of metric file names, in order from most relevant to least relevant.

```
# Example output:
# ['LevenshteinDistance.md', 'UniEvalDialogue.md', 'CIDEr.md', ...] # length 20
```

You can include as much reasoning as you need, but please make your final answer a single list of metric file names, in order from most relevant to least relevant and in a code block.

Additionally, please ensure that the list is all on one line, and that there are no newlines or extra spaces in the list itself (match the example output format).  Thanks!
"""

doc_strings = "\n".join([f"======\n{doc_id}\n======\n{doc}\n======\n" for doc_id, doc in zip(doc_ids, documents)])

for task in tasks:
    task_name = task.name
    targets = task.get_target_columns()

    for target in targets:
        print(f"Task: {task_name}, Target: {target}")

        task_description = task_descriptions[task_name]
        prompt = PROMPT_TEMPLATE.format(
            task_description=task_description,
            target=target,
            documents=doc_strings
        )
        with open(f"{output_folder}/{task_name}_{target}.txt", "w") as f:
            f.write(prompt)
        print(f"Prompt saved to {output_folder}/{task_name}_{target}.txt")