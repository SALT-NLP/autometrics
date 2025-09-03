import os
import dspy

from autometrics.dataset.datasets.simplification.simplification import SimpDA
from autometrics.metrics.reference_based.BLEU import BLEU
from autometrics.metrics.reference_based.SARI import SARI
from autometrics.metrics.reference_based.ROUGE import ROUGE
from autometrics.aggregator.regression.PLS import PLS
from autometrics.util.report_card import generate_metric_report_card


def main():
    dataset = SimpDA()
    # small subset for speed
    train = dataset.get_subset(80, seed=42)
    eval_ds = dataset.get_subset(60, seed=7)

    metrics = [BLEU(), SARI(), ROUGE()]

    # Compute metrics on train set
    for m in metrics:
        train.add_metric(m, update_dataset=True)

    # Train simple PLS regression as aggregator over all metric columns
    reg = PLS(name="Autometrics_Regression_SimpDA", description="PLS aggregator", dataset=train, input_metrics=metrics)
    reg.learn(train, target_column=train.get_target_columns()[0])

    # Prepare a lightweight LM for summary and robustness (user may configure OPENAI or local)
    # Use LiteLLM proxy for Qwen3-32B hosted at sphinx7.stanford.edu:8219
    lm = dspy.LM("litellm_proxy/Qwen/Qwen3-32B", api_base="http://sphinx7.stanford.edu:8219/v1", api_key="None")

    out_path = os.path.join("artifacts", "report_minimal_simpda.html")
    art = generate_metric_report_card(
        regression_metric=reg,
        metrics=metrics,
        target_measure=train.get_target_columns()[0],
        eval_dataset=eval_ds,
        train_dataset=train,
        lm=lm,
        output_path=out_path,
        verbose=True,
    )

    print(f"Report saved to: {art.get('path')} (length={len(art.get('html',''))})")


if __name__ == "__main__":
    main()


