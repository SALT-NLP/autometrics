from autometrics.metrics.reference_based.ReferenceBasedMetric import ReferenceBasedMetric
from sacrebleu.metrics import CHRF as chrf

class CHRF(ReferenceBasedMetric):

    def __init__(self, name="CHRF", description="chrF++ is a metric for evaluating machine translation quality that uses character and word n-gram F-scores to assess similarity between translations and references. It captures both fine-grained character-level details and word-level structure, making it effective for languages with rich morphology."):
        super().__init__(name, description)
        self.metric = chrf()

    def calculate(self, input, output, references=None, **kwargs):
        """
        Calculate the metric
        """
        if references is None:
            references = []

        output = [output]
        references = [[r] for r in references]

        return self.metric.corpus_score(output, references).score
