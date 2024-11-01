from autometrics.metrics.reference_based.ReferenceBasedMetric import ReferenceBasedMetric
from sacrebleu.metrics import BLEU as bleu

class BLEU(ReferenceBasedMetric):

    def __init__(self, name="BLEU", description="BLEU compares the n-grams of the candidate with the n-grams of the reference."):
        super().__init__(name, description)
        self.metric = bleu()

    def calculate(self, input, output, references=None, **kwargs):
        """
        Calculate the metric
        """
        if references is None:
            references = []

        output = [output]
        references = [[r] for r in references]

        return self.metric.corpus_score(output, references).score
