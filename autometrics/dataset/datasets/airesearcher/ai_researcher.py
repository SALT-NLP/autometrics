from autometrics.dataset.Dataset import Dataset
import pandas as pd
from typing import Literal
from autometrics.metrics.dummy import DummyMetric
class AI_Researcher(Dataset):
    def __init__(self, path='./autometrics/dataset/datasets/airesearcher/ideas_aggregated.csv'): # Path to the dataset: './autometrics/dataset/datasets/cogym/cogym.csv'
        # userId,sessionId,modelName,task,query,createdAt,finishedAt,agentRating,communicationRating,outcomeRating,agentFeedback,finished,bookmarked,agentType,conversation,formatted_conversation,outcome
        df = pd.read_csv(path)

        df.drop(columns=['topic'], inplace=True)

        target_columns = ['novelty_score', 'feasibility_score', 'effectiveness_score', 'excitement_score', 'overall_score']

        ignore_columns = ["novelty_rationale", "feasibility_rationale", "effectiveness_rationale", "excitement_rationale", "overall_rationale", 'confidence_score']
        metric_columns = []

        name = "airesearcher"

        data_id_column = "idea_id"
        model_id_column = "condition"
        input_column = "Title / Filename"
        output_column = "Content"
        reference_columns = []

        metrics = [DummyMetric(col) for col in metric_columns]

        task_description = "You are an expert researcher in Natural Language Processing. Now I want you to help me brainstorm some new research project ideas on the topic of: {topic_description}. You should generate {ideas_n} different ideas on this topic. Try to be creative and diverse in the idea generation, and do not repeat any similar ideas. We are targeting the EMNLP 2024 conference (The 2024 Conference on Empirical Methods in Natural Language Processing), and you should aim for timely and impactful new ideas that can potentially win best paper awards at EMNLP. EMNLP 2024 invites the submission of papers featuring substantial, original, and unpublished research on empirical methods for Natural Language Processing. The type of contribution can include: formulate new problems, propose new methods that outperform existing baselines, construct new datasets or benchmarks, propose new evaluation metrics, propose novel applications of NLP, conduct empirical analysis, or any other novel contribution that advances the field of NLP. Note that we do not take survey or position papers - there has to be some computational experiments involved. Each idea should be described as: (1) Problem: State the problem statement, which should be closely related to the given topic and within the scope of NLP research. (2) Existing Work: Mention the most relevant existing work. (3) Motivation: Explain the inspiration of the proposed study and why it would work well or be important to study. (4) Proposed Study: Propose your new method or analysis or benchmark and describe it in detail. The proposal should be maximally different from all existing work and baselines, and be more advanced and effective than the baselines. You should be as creative as possible, we love unhinged ideas that sound crazy. This should be the most detailed section of the proposal. (5) Experiment Plan: Specify the hypotheses, experiment steps, baselines, evaluation metrics, and/or any other relevant details. You can follow these examples to get a sense of how the ideas should be formatted (but don't borrow the ideas themselves): {examples} You should make sure to come up with your own novel and different ideas for the specified problem: {topic_description}"

        super().__init__(df, target_columns, ignore_columns, metric_columns, name, data_id_column, model_id_column, input_column, output_column, reference_columns, metrics, task_description=task_description)