from autometrics.experiments.experiment import Experiment
import dspy
from typing import Union, Literal, Callable
from autometrics.util.format import get_default_formatter
from autometrics.dataset.Dataset import Dataset
from autometrics.metrics.MultiMetric import MultiMetric
import pandas as pd
from autometrics.experiments.results import TabularResult, FigureResult, PlotlyResult
import litellm
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import hashlib
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
import matplotlib.pyplot as plt
import plotly.graph_objects as go

litellm.suppress_debug_info = True

class GeneratePerturbationStrategies(dspy.Signature):
    """You will be given:  
• A Task description  
• A Dimension to prioritize when perturbing outputs  
• The Example Input, optional Example Reference, and Example Output  

Instructions:  
Your primary focus should be on degrading performance along the specified Dimension.  
1. Begin with a rich reasoning paragraph (3–5 sentences) that explores a variety of ways to subtly degrade model outputs. Do **not** reference the specific example.  
2. Under the heading **Strategies:**, list **1–3** numbered, high-level perturbation strategies.  
   - Each strategy should be a short phrase (5–15 words) naming the category of change, followed by one concise sentence of abstract explanation.  
   - Do **not** include concrete rewrites, instance-specific examples, or example sentences.  

Task: Given a complicated original sentence, simplify it so a broader audience can easily understand it.  
Example Input: after the jerilderie raid, the gang laid low for 16 months evading capture.  
Example Reference: after the jerilderie raid, the gang laid low for 16 months avoiding capture.  
Example Output: after the jerilderie raid, the gang successfully hid for 16 months.  
Dimension: Meaning Preservation"""
    task: str = dspy.InputField(description="The task that the model was originally trying to complete")
    example_sets: list[str] = dspy.InputField(description="Example inputs, outputs, and (optionally) references showcasing the model's performance on the task")
    dimension: str = dspy.InputField(description="The dimension to prioritize for the perturbation (this should be the aspect of the model output that is most impacted by the perturbation)")
    perturbation_strategies: list[str] = dspy.OutputField(description="A list of perturbation strategies that can be used to test the robustness of the model")


class PerturbWorse(dspy.Signature):
    """You will be given:  
    • A Task description  
    • A Dimension to prioritize when perturbing outputs  
    • The Example Input, optional Example Reference, and Model Output  
    • A perturbation_strength value ("subtle" or "obvious")  
    • A list of perturbation_strategies to apply  

Instructions:  
Your goal is to apply each strategy to the Model Output and produce a degraded version that specifically harms performance along the given Dimension, using the specified strength.  
Under the heading **Perturbed Outputs:**, return exactly one perturbed output per strategy.  
    - For **subtle** strength, introduce minimal distortion.  
    - For **obvious** strength, introduce more pronounced degradation.  
Do **not** include any reasoning, explanations, or examples -- only the perturbed text."""
    task: str = dspy.InputField(description="The task that the model was originally trying to complete")
    dimension: str = dspy.InputField(description="The dimension to prioritize for the perturbation (this should be the aspect of the model output that is most impacted by the perturbation)")
    input: str = dspy.InputField(description="The input provided to the model")
    references: Union[list[str], None] = dspy.InputField(description="The references of good outputs (may be None)")
    model_output: str = dspy.InputField(description="The output produced by the model")
    perturbation_strength: Literal["subtle", "obvious"] = dspy.InputField(description="The strength of the perturbation (subtle or obvious)")
    perturbation_strategies: list[str] = dspy.InputField(description="The perturbation strategies to use")
    perturbed_outputs: list[str] = dspy.OutputField(description="Perturbed text that is worse than the original model output.  Produce one perturbed output per strategy.")

class PerturbSame(dspy.Signature):
    """You will be given:
    • A Task description  
    • A Dimension to preserve when perturbing outputs  
    • The Example Input, optional Example Reference, and Model Output  
    • A perturbation_strength value ("subtle" or "obvious")  

Instructions:
Apply a perturbation to the Model Output that **maintains** performance on the specified Dimension.
Under the heading **Perturbed Output:** return exactly one string:
    - For **subtle** strength, apply a minimal change that does not impair the target Dimension.
    - For **obvious** strength, apply a more noticeable change that still keeps the target Dimension intact.
Some examples of types of perturbations would include: rephrasing, reordering, replacing words with synonyms, stylistic changes, etc. that do not impair the target Dimension.
If any change would harm the specified Dimension, simply return the original Model Output.
After producing your original plan/reasoning do **not** include any more reasoning, explanations, or examples -- only the perturbed text."""
    task: str = dspy.InputField(description="The task that the model was originally trying to complete")
    input: str = dspy.InputField(description="The input provided to the model")
    references: Union[list[str], None] = dspy.InputField(description="The references of good outputs (may be None)")
    model_output: str = dspy.InputField(description="The output produced by the model")
    perturbation_strength: Literal["subtle", "obvious"] = dspy.InputField(description="The strength of the perturbation (subtle or obvious)")
    dimension: str = dspy.InputField(description="The aspect of the model output that MUST be preserved in quality")
    perturbed_output: str = dspy.OutputField(description="Perturbed text that preserves performance along the given Dimension.")

class ProducePerturbations(dspy.Module):

    def __init__(self, num_examples: int = 3, formatter: Callable = None, max_workers: int = None):
        self.generate_perturbation_strategies: GeneratePerturbationStrategies = dspy.ChainOfThought(GeneratePerturbationStrategies)
        self.perturb_worse: PerturbWorse = dspy.Predict(PerturbWorse)
        self.perturb_same: PerturbSame = dspy.ChainOfThought(PerturbSame)
        self.num_examples = num_examples
        self.formatter = formatter
        self.max_workers = max_workers if max_workers is not None else 1

    def forward(self, task: str, dimension: str, dataset: Dataset):
        # --- setup formatter and examples ---
        if self.formatter is None:
            self.formatter = get_default_formatter(dataset)

        df = dataset.get_dataframe()
        sampled_rows = df.sample(self.num_examples) if self.num_examples < len(df) else df
        formatted_rows = [ self.formatter(row) for row in sampled_rows.iterrows() ]

        # generate your list of strategies once
        perturbation_strategies = self.generate_perturbation_strategies(
            task=task,
            dimension=dimension,
            example_sets=formatted_rows
        ).perturbation_strategies

        # extract column names once
        input_col = dataset.get_input_column()
        output_col = dataset.get_output_column()
        ref_cols = dataset.get_reference_columns()

        # convert DataFrame into list of plain dicts for easy pickling
        records = df.to_dict('records')

        # prepare accumulators
        overall_worse_subtle, overall_worse_obvious = [], []
        overall_same_subtle, overall_same_obvious   = [], []

        # define worker
        def _process(record):
            inp = record[input_col]
            refs = [record[c] for c in ref_cols]
            out = record[output_col]

            res_worse_subtle = self.perturb_worse(
                task=task,
                dimension=dimension,
                input=inp,
                references=refs,
                model_output=out,
                perturbation_strength="subtle",
                perturbation_strategies=perturbation_strategies
            ).perturbed_outputs

            res_worse_obvious = self.perturb_worse(
                task=task,
                dimension=dimension,
                input=inp,
                references=refs,
                model_output=out,
                perturbation_strength="obvious",
                perturbation_strategies=perturbation_strategies
            ).perturbed_outputs

            res_same_subtle = self.perturb_same(
                task=task,
                dimension=dimension,
                input=inp,
                references=refs,
                model_output=out,
                perturbation_strength="subtle"
            ).perturbed_output

            res_same_obvious = self.perturb_same(
                task=task,
                dimension=dimension,
                input=inp,
                references=refs,
                model_output=out,
                perturbation_strength="obvious"
            ).perturbed_output

            return res_worse_subtle, res_worse_obvious, res_same_subtle, res_same_obvious

        # spin up thread‐pool and collect results
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for worse_subtle, worse_obvious, same_subtle, same_obvious in tqdm(
                executor.map(_process, records),
                total=len(records),
                desc="perturbing examples"
            ):
                overall_worse_subtle.append(worse_subtle)
                overall_worse_obvious.append(worse_obvious)
                overall_same_subtle.append(same_subtle)
                overall_same_obvious.append(same_obvious)

        return {
            "perturbed_worse_subtle": overall_worse_subtle,
            "perturbed_worse_obvious": overall_worse_obvious,
            "perturbed_same_subtle": overall_same_subtle,
            "perturbed_same_obvious": overall_same_obvious,
            "strategies": perturbation_strategies
        }
# --- ANALYSIS HELPERS -------------------------------------------------------

# 1) Bump Matplotlib font sizes globally
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
})

def _get_metric_cols(metric_objs):
    """Expand Metric and MultiMetric into a list of column names."""
    cols = []
    for m in metric_objs:
        if hasattr(m, 'get_submetric_names'):
            cols.extend(m.get_submetric_names())
        else:
            cols.append(m.get_name())
    return cols

def _normalize_df(df, cols):
    """Min-max normalize specified columns to [0,1]."""
    df_norm = df.copy()
    df_norm[cols] = MinMaxScaler().fit_transform(df_norm[cols])
    return df_norm

def _get_tukey_clusters(groups, tukey):
    """Union-find clusters of groups that Tukey HSD does NOT reject."""
    parent = {g: g for g in groups}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for row in tukey.summary().data[1:]:
        g1, g2, *_, reject = row
        if not reject:
            union(g1, g2)
    clusters = {}
    for g in groups:
        root = find(g)
        clusters.setdefault(root, []).append(g)
    return clusters

def _do_anova_tukey(df_norm, col, dimension, results):
    """Run ANOVA + Tukey HSD, save tables, return the Tukey object."""
    # ANOVA
    model = ols(f"{col} ~ C(group)", data=df_norm).fit()
    anova_tbl = sm.stats.anova_lm(model, typ=2)
    results[f"{dimension}/{col}/anova"] = TabularResult(anova_tbl)
    # Tukey HSD
    mc = MultiComparison(df_norm[col], df_norm['group'])
    tukey = mc.tukeyhsd(alpha=0.05)
    tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
    results[f"{dimension}/{col}/tukey"] = TabularResult(tukey_df)
    return tukey

def _plot_bar_mpl(df_norm, col, dimension, tukey, results):
    """Static bar chart colored by Tukey clusters."""
    groups = ['original','same_subtle','same_obvious','worse_subtle','worse_obvious']
    means = df_norm.groupby('group')[col].mean().reindex(groups)
    ci95  = 1.96 * df_norm.groupby('group')[col].sem().reindex(groups)
    clusters = _get_tukey_clusters(groups, tukey)
    roots   = list(clusters.keys())
    cmap    = plt.get_cmap('tab10')
    colors  = [cmap(roots.index(next(r for r,m in clusters.items() if g in m))) for g in groups]

    fig, ax = plt.subplots(figsize=(8,6))
    x = np.arange(len(groups))
    ax.bar(x, means, yerr=ci95, capsize=5, color=colors)
    ax.axhline(means['original'], ls='--', color='gray')
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace('_',' ').title() for g in groups], rotation=30)
    ax.set_ylabel(f"Normalized {col}")
    ax.set_title(f"{col} vs Perturbation Type ({dimension.capitalize()})")
    plt.tight_layout()
    results[f"{dimension}/{col}/bar"] = FigureResult(fig)

def _plot_bar_plotly(df_norm, col, dimension, tukey, results):
    """Interactive Plotly bar chart, no legend, cluster-colored."""
    groups = ['original','same_subtle','same_obvious','worse_subtle','worse_obvious']
    means = df_norm.groupby('group')[col].mean().reindex(groups)
    ci95  = 1.96 * df_norm.groupby('group')[col].sem().reindex(groups)
    clusters = _get_tukey_clusters(groups, tukey)
    roots = list(clusters.keys())
    cmap = plt.get_cmap('tab10')
    group_color = {}
    for idx, root in enumerate(roots):
        for g in clusters[root]:
            r, g_, b, a = [int(255*x) for x in cmap(idx)]
            group_color[g] = f"rgba({r},{g_},{b},{a})"

    fig = go.Figure()
    for g in groups:
        fig.add_trace(go.Bar(
            x=[g.replace('_',' ').title()],
            y=[means[g]],
            error_y=dict(type='data', array=[ci95[g]]),
            marker_color=group_color[g],
            showlegend=False
        ))
    # dashed line at original mean
    fig.add_shape(dict(
        type='line', x0=-0.5, x1=len(groups)-0.5,
        y0=means['original'], y1=means['original'],
        line=dict(color='gray', dash='dash')
    ))
    fig.update_layout(
        title=f"{col} vs Perturbation Type ({dimension.capitalize()})",
        xaxis_title="Perturbation Type",
        yaxis_title=f"Normalized {col}",
        barmode='group',
        font=dict(size=16),
        showlegend=False
    )
    results[f"{dimension}/{col}/bar_interactive"] = PlotlyResult(fig)

def _build_summary(df_norm, cols):
    """Compute sensitivity & stability summary for given cols."""
    recs = []
    orig = df_norm[df_norm['group']=='original']
    for col in cols:
        for g in ['worse_subtle','worse_obvious']:
            part   = df_norm[df_norm['group']==g]
            merged = pd.merge(part[['sample_id',col]], orig[['sample_id',col]],
                              on='sample_id', suffixes=('_p','_o'))
            sens   = (merged[f"{col}_o"] - merged[f"{col}_p"]).mean()
            recs.append({'metric':col, f'sensitivity_{g}': sens})
        for g in ['same_subtle','same_obvious']:
            part   = df_norm[df_norm['group']==g]
            merged = pd.merge(part[['sample_id',col]], orig[['sample_id',col]],
                              on='sample_id', suffixes=('_p','_o'))
            stab   = 1 - np.abs(merged[f"{col}_o"] - merged[f"{col}_p"]).mean()
            recs.append({'metric':col, f'stability_{g}': stab})
    return pd.DataFrame(recs).groupby('metric').first().reset_index()

def _plot_scatter_mpl(summary_df, dimension, results):
    """Static scatter of avg stability vs avg sensitivity."""
    fig, ax = plt.subplots(figsize=(8,6))
    for _, r in summary_df.iterrows():
        x = np.mean([r['stability_same_subtle'], r['stability_same_obvious']])
        y = np.mean([r['sensitivity_worse_subtle'], r['sensitivity_worse_obvious']])
        ax.scatter(x, y, s=200, marker='o', color='tab:blue', zorder=5)
        ax.annotate(r['metric'], xy=(x,y), xytext=(-4,4),
                    textcoords='offset points', fontsize=14, ha='right', va='bottom')
    ax.set_xlabel("Avg Stability")
    ax.set_ylabel("Avg Sensitivity")
    ax.set_title(f"Stability vs Sensitivity ({dimension.capitalize()})")
    plt.tight_layout()
    results[f"{dimension}/scatter"] = FigureResult(fig)

def _plot_scatter_plotly(summary_df, dimension, results):
    """Interactive scatter of avg stability vs avg sensitivity with legend."""
    fig = go.Figure()
    for _, r in summary_df.iterrows():
        x = np.mean([r['stability_same_subtle'], r['stability_same_obvious']])
        y = np.mean([r['sensitivity_worse_subtle'], r['sensitivity_worse_obvious']])
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            name=r['metric'],
            text=[r['metric']],
            textposition='top left',
            marker=dict(size=14)
        ))
    fig.update_layout(
        title=f"Stability vs Sensitivity ({dimension.capitalize()})",
        xaxis_title="Average Stability",
        yaxis_title="Average Sensitivity",
        font=dict(size=16)
    )
    results[f"{dimension}/scatter_interactive"] = PlotlyResult(fig)

def analyze_and_plot(df: pd.DataFrame, metric_objs: list, dimension: str, results: dict):
    """
    Orchestrates normalization, ANOVA/Tukey, bar & scatter plots (static + interactive).
    """
    cols    = _get_metric_cols(metric_objs)
    df_norm = _normalize_df(df, cols)
    # 1) ANOVA/Tukey + bar plots
    for col in cols:
        tukey = _do_anova_tukey(df_norm, col, dimension, results)
        _plot_bar_mpl(df_norm, col, dimension, tukey, results)
        _plot_bar_plotly(df_norm, col, dimension, tukey, results)
    # 2) Summary table + scatter
    summary_df = _build_summary(df_norm, cols)
    results[f"{dimension}/sens_stab"] = TabularResult(summary_df)
    _plot_scatter_mpl(summary_df, dimension, results)
    _plot_scatter_plotly(summary_df, dimension, results)

class RobustnessExperiment(Experiment):

    def _produce_perturbation_scores(self, dataset: Dataset, perturbations: dict[str, list[str]]) -> pd.DataFrame:
        worse_subtle, worse_obvious, same_subtle, same_obvious, strategies = perturbations["perturbed_worse_subtle"], \
                                                                                     perturbations["perturbed_worse_obvious"], \
                                                                                     perturbations["perturbed_same_subtle"], \
                                                                                     perturbations["perturbed_same_obvious"], \
                                                                                     perturbations["strategies"]
        
        inputs = dataset.get_dataframe()[dataset.get_input_column()].tolist()
        reference_columns = dataset.get_reference_columns()
        
        # Create a dictionary where each key is a reference column and value is the list of values

        inputs_structured = [[inputs[i]] * len(strategies) for i in range(len(inputs))]
        inputs_structured = [item for sublist in inputs_structured for item in sublist]

        data = {
            "input": inputs_structured + inputs_structured + inputs + inputs,
            "model_output": [],
            "strategy": (strategies * len(worse_subtle)) + (strategies * len(worse_obvious)) + (["same_subtle"] * len(same_subtle)) + (["same_obvious"] * len(same_obvious)),
            "group": ["worse_subtle"] * len(worse_subtle) * len(strategies) + ["worse_obvious"] * len(worse_obvious) * len(strategies) + ["same_subtle"] * len(same_subtle) + ["same_obvious"] * len(same_obvious)
        }
        
        # Add each reference column to the data dictionary, duplicated amt_to_eval times
        for ref_col in reference_columns:
            ref_values = dataset.get_dataframe()[ref_col].tolist()

            ref_values_structured = [[ref_values[i]] * len(strategies) for i in range(len(ref_values))]
            ref_values_structured = [item for sublist in ref_values_structured for item in sublist]

            data[ref_col] = ref_values_structured + ref_values_structured + ref_values + ref_values

        data["model_output"].extend([item for sublist in worse_subtle for item in sublist])
        data["model_output"].extend([item for sublist in worse_obvious for item in sublist])
        data["model_output"].extend(same_subtle)
        data["model_output"].extend(same_obvious)

        df = pd.DataFrame(data)

        for metric in self.metrics:
            original_values = dataset.get_metric_values(metric)
            true_outputs = dataset.get_dataframe()[dataset.get_output_column()]

            results = metric.calculate_batched(df["input"], df["model_output"], [[df[ref_col].iloc[i] for ref_col in reference_columns] for i in range(len(df))])

            # Check if the metric is a multi-metric
            if isinstance(results, (list, tuple)) and isinstance(metric, MultiMetric):
                for i, submetric_name in enumerate(metric.get_submetric_names()):
                    data[submetric_name] = list(results[i])
                    data[submetric_name].extend(original_values[submetric_name])
            else:
                data[metric.get_name()] = results
                data[metric.get_name()].extend(original_values)

        data["input"].extend(inputs)
        for ref_col in reference_columns:
            data[ref_col].extend(dataset.get_dataframe()[ref_col].tolist())
        data["model_output"].extend(true_outputs)
        data["strategy"].extend([["original"]] * len(inputs))
        data["group"].extend(["original"] * len(inputs))

        df = pd.DataFrame(data)
        return df

    def run(self, print_results: bool = False, num_demonstration_examples: int = 3, max_eval_examples: int = 30, max_workers: int = 8):

        test_dataset = self.test_dataset
        if max_eval_examples < len(test_dataset.get_dataframe()):
            test_dataset = test_dataset.get_subset(max_eval_examples, seed=self.seed)

        produce_perturbations = ProducePerturbations(num_examples=num_demonstration_examples, max_workers=max_workers)

        if self.kwargs.get("lm"):
            self.lm = self.kwargs.get("lm")
        else:
            self.lm = dspy.settings.lm

        with dspy.settings.context(lm=lm):
            for column in test_dataset.get_target_columns():
                # First, generate the perturbations
                perturbations = produce_perturbations(task=test_dataset.get_task_description(), dimension=column, dataset=test_dataset)
                df = self._produce_perturbation_scores(test_dataset, perturbations)
                self.results[f"{column}/full_table"] = TabularResult(df)

                if print_results:
                    print(df)

                df['sample_id'] = df['input'].str.strip().str.lower().apply(lambda x: hashlib.md5(x.encode()).hexdigest())
                analyze_and_plot(df, self.metrics, column, self.results)

if __name__ == "__main__":

    from autometrics.dataset.datasets.simplification.simplification import SimpDA
    from autometrics.metrics.reference_based.BLEU import BLEU
    from autometrics.metrics.reference_based.SARI import SARI
    import os


    lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv("OPENAI_API_KEY"))
    dspy.configure(lm=lm)

    dataset = SimpDA()

    experiment = RobustnessExperiment(
        name="Robustness Experiment",
        description="An experiment to test the robustness of the model",
        metrics=[BLEU(), SARI()],
        output_dir="outputs/robustness",
        dataset=dataset
    )

    experiment.run(print_results=True)

    experiment.save_results()