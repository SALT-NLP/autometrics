from autometrics.dataset.datasets import Design2Code
from autometrics.util.analysis import display_top_5_metrics_by_validation, get_top_metric_by_validation, plot_metric_target_scatterplot
from autometrics.aggregator.regression import Ridge
from autometrics.generator.LLMJudgeProposer import LLMJudgeProposer
import dspy
import litellm
from autometrics.metrics.MetricBank import all_metrics

litellm.suppress_debug_info = True

design_prompt = """You are an expert web developer who specializes in HTML and CSS. A user will provide you with a screenshot of a webpage. You need to return a single html file that uses HTML and CSS to reproduce the given website. Include all CSS code in the HTML file itself. If it involves any images, use "rick.jpg" as the placeholder. Some images on the webpage are replaced with a blue rectangle as the placeholder, use "rick.jpg" for those as well. Do not hallucinate any dependencies to external files. You do not need to include JavaScript scripts for dynamic interactions. Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout. Respond with the content of the HTML+CSS file."""

task_prompt = design_prompt

# %%
dataset = Design2Code()

# %%
train, dev, test = dataset.get_splits(train_ratio=0.2, val_ratio=0.2, seed=42, max_size=1000)

print("Train size:", len(train.dataframe), "Validation size:", len(dev.dataframe), "Test size:", len(test.dataframe))

# %%
llama33_70b = dspy.LM("litellm_proxy/meta-llama/Meta-Llama-3.3-70b-Instruct", api_base="http://future-hgx-2:7410/v1", api_key="None")

dspy.settings.configure(lm=llama33_70b)

# %%
generator = LLMJudgeProposer(train_dataset=train, task_description=task_prompt, proposer_model=llama33_70b, judge_model=llama33_70b)

# %%
print(llama33_70b.model)

for target_column in dataset.target_columns:
    train.add_metrics(all_metrics)
    dev.add_metrics(all_metrics)
    test.add_metrics(all_metrics)

df = display_top_5_metrics_by_validation(dev, test, True)
print(df)
df.to_csv("outputs/" + dataset.name + "_top_metrics.csv")

# %%
new_metrics = all_metrics

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
dataset.get_dataframe().to_csv("outputs/" + dataset.name + "_all.csv")