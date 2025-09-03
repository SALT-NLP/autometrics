import os
import dspy

from autometrics.autometrics import Autometrics
from autometrics.dataset.datasets.simplification.simplification import SimpDA
from autometrics.metrics.reference_based.BLEU import BLEU
from autometrics.metrics.reference_based.SARI import SARI
from autometrics.metrics.reference_based.ROUGE import ROUGE
from autometrics.recommend.MetricRecommender import MetricRecommender


def main():
    # Configure small LMs (user should set keys/env as needed)
    # Use LiteLLM proxy for Qwen3-32B hosted at sphinx7.stanford.edu:8219
    gen_lm = dspy.LM("litellm_proxy/Qwen/Qwen3-32B", api_base="http://sphinx7.stanford.edu:8123/v1", api_key="None")
    judge_lm = gen_lm

    am = Autometrics(
        metric_generation_configs={
            "llm_judge": {"metrics_per_trial": 2, "description": "Basic LLM Judge"}
        },
        metric_bank=[BLEU, SARI, ROUGE],
        merge_generated_with_bank=False,
        enable_parallel_evaluation=True,
        max_parallel_workers=8,
    )

    dataset = SimpDA()
    train = dataset.get_subset(120, seed=42)
    eval_ds = dataset.get_subset(80, seed=24)

    out = am.run(
        dataset=train,
        target_measure=train.get_target_columns()[0],
        generator_llm=gen_lm,
        judge_llm=judge_lm,
        num_to_retrieve=5,
        num_to_regress=3,
        regenerate_metrics=False,
        eval_dataset=eval_ds,
        report_output_path=os.path.join("artifacts", "report_pipeline.html"),
    )

    print("Top metrics:", [m.get_name() for m in out["top_metrics"]])
    print("Report saved:", out.get("report_card_path"))


if __name__ == "__main__":
    main()


