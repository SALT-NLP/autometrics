from autometrics.metrics.reference_based.ReferenceBasedMetric import ReferenceBasedMetric
from sacrebleu.metrics import TER as ter

class TER(ReferenceBasedMetric):

    def __init__(self, name="TER", description="TER (Translation Edit Rate) is a metric that measures the number of edits needed to transform a system output into a reference translation. It quantifies translation quality by counting insertions, deletions, substitutions, and shifts, with lower scores indicating better translations."):
        super().__init__(name, description)
        self.metric = ter()

    def calculate(self, input, output, references=None, **kwargs):
        """
        Calculate the metric
        """
        if references is None:
            references = []

        output = [output]
        references = [[r] for r in references]

        return self.metric.corpus_score(output, references).score
