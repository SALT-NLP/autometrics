from autometrics.dataset.datasets import RealHumanEval
from autometrics.util.analysis import display_top_5_metrics_by_validation, get_top_metric_by_validation, plot_metric_target_scatterplot
from autometrics.aggregator.regression import Ridge
from autometrics.generator.LLMJudgeProposer import LLMJudgeProposer
import dspy
import litellm
from autometrics.metrics.MetricBank import reference_free_metrics

litellm.suppress_debug_info = True

humaneval_prompt = """You are an expert Python programmer, be helpful to the user and return code only in Python."""

task_prompt = humaneval_prompt

# %%
dataset = RealHumanEval()

# %%
train, dev, test = dataset.get_splits(train_ratio=0.2, val_ratio=0.2, seed=42, max_size=1000)

# %%
llama33_70b = dspy.LM("litellm_proxy/meta-llama/Meta-Llama-3.3-70b-Instruct", api_base="http://localhost:7410/v1", api_key="local", model_type='chat')

dspy.settings.configure(lm=llama33_70b)

# %%
generator = LLMJudgeProposer(train_dataset=train, task_description=task_prompt, proposer_model=llama33_70b, judge_model=llama33_70b)

# %%
print(llama33_70b.model)

for target_column in dataset.target_columns:
    train.add_metrics(reference_free_metrics)
    dev.add_metrics(reference_free_metrics)
    test.add_metrics(reference_free_metrics)

df = display_top_5_metrics_by_validation(dev, test, True)
print(df)
df.to_csv("outputs/" + dataset.name + "_top_metrics.csv")

# %%
new_metrics = reference_free_metrics

for target_column in dataset.target_columns:
    new_metrics.extend(generator.generate(train, target_column))

# %%
# Condense the metrics that have duplicate names
new_metrics_names = set()
new_metrics_final = []
for metric in new_metrics:
    if metric.name not in new_metrics_names:
        new_metrics_names.add(metric.name)
        new_metrics_final.append(metric)

# %%
train.add_metrics(new_metrics_final)
dev.add_metrics(new_metrics_final)
test.add_metrics(new_metrics_final)

# %%
df = display_top_5_metrics_by_validation(dev, test, True)
print(df)
df.to_csv("outputs/" + dataset.name + "_top_metrics_dspy.csv")

# %%
from tqdm import tqdm
for target in tqdm(dataset.get_target_columns()):
    aggregator = Ridge(dataset=train, name=f'Ridge_{target}_llm')
    aggregator.ensure_dependencies(train)
    aggregator.ensure_dependencies(dev)
    aggregator.ensure_dependencies(test)

# %%
train.get_metric_columns()

# %%
for target in tqdm(dataset.get_target_columns()):
    aggregator = Ridge(dataset=train, name=f'Ridge_{target}_llm')
    aggregator.learn(train, target)
    aggregator.predict(train)
    aggregator.predict(dev)
    aggregator.predict(test)

# %%
df = display_top_5_metrics_by_validation(dev, test, True)
print(df)
df.to_csv("outputs/" + dataset.name + "_top_metrics_dspy_regression.csv")

# all computed values
train.get_dataframe().to_csv("outputs/" + dataset.name + "_train.csv")
dev.get_dataframe().to_csv("outputs/" + dataset.name + "_dev.csv")
test.get_dataframe().to_csv("outputs/" + dataset.name + "_test.csv")