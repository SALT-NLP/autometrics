from autometrics.dataset.datasets import EvalGenProduct, EvalGenMedical
from autometrics.util.analysis import display_top_5_metrics_by_validation, get_top_metric_by_validation, plot_metric_target_scatterplot
from autometrics.aggregator.regression import Ridge
from autometrics.generator.LLMJudgeProposer import LLMJudgeProposer
import dspy
import litellm
from autometrics.metrics.MetricBank import reference_free_metrics

litellm.suppress_debug_info = True

product_prompt = """You are an expert copywriter. You need to write an e-commerce product description based on the product details and customer reviews. Your description should be SEO-optimized. It should use an active voice and include the product's features, benefits, unique selling points without overpromising, and a call to action for the buyer. Benefits describe how product features will work for the buyer, addressing exactly how the product will improve their lives. Clearly distinguish between features (e.g., lightweight, USB-chargeable) and benefits (e.g., convenience, nutritious drinks on-the-go). Don't mention weaknesses of the product or use generic or repetitive language. Don't make up review text or quotes. Don't include any links. Don't cite the reviews too heavily. Divide your description into readable chunks divided by relevant subheadings. Keep your description around 200 words, no more than 300, in Markdown format.\n\n"""
medical_prompt = """You are extracting insights from some medical records. The records contain a medical note and a dialogue between a doctor and a patient. You need to extract values for the following: Chief complaint, History of present illness, Physical examination, Symptoms experienced by the patient, New medications prescribed or changed, including dosages (N/A if not provided), and Follow-up instructions (N/A if not provided). Your answer should not include any personal identifiable information (PII) such as name, age, gender, or ID. Use "the patient" instead of their name, for example. Return your answer as a bullet list, where each bullet is formatted like `chief complaint: xx.` If there is no value for the key, the value should be `N/A`. Keep your response around 150 words (you may have to summarize some extracted values to stay within the word limit)."""

task_prompt = product_prompt

# %%
dataset = EvalGenProduct()

# %%
train, dev, test = dataset.get_splits(train_ratio=0.2, val_ratio=0.2, seed=42, max_size=1000)

# %%
llama33_70b = dspy.LM("litellm_proxy/meta-llama/Meta-Llama-3.3-70b-Instruct", api_base="http://future-hgx-1.stanford.edu:7410/v1", api_key="local", model_type='chat')

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