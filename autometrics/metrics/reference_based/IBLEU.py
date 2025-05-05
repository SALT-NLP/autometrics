from sacrebleu.metrics import BLEU as bleu
from autometrics.metrics.reference_based.ReferenceBasedMetric import ReferenceBasedMetric

class IBLEU(ReferenceBasedMetric):
    """iBLEU score combining BLEU similarity to references and self-BLEU diversity penalty."""

    def __init__(self,
                 name: str = "iBLEU",
                 description: str = "iBLEU metric combining BLEU score to references and a self-BLEU penalty for diversity.",
                 alpha: float = 0.9):
        super().__init__(name, description)
        self.metric = bleu()
        self.alpha = alpha

    def calculate(self, input: str, output: str, references=None, alpha: float = None, **kwargs) -> float:
        """
        Calculate the iBLEU score for a hypothesis.

        Args:
            input: Source sentence (string).
            output: Candidate translation (string).
            references: List of reference translation strings.
            alpha: Weight for reference BLEU (overrides default).
        Returns:
            A float iBLEU score: alpha * BLEU(refs, cand) - (1-alpha) * BLEU(src, cand).
        """
        if references is None:
            references = []
        # determine alpha
        alpha_val = alpha if alpha is not None else self.alpha

        # prepare streams for sacreBLEU
        sys_stream = [output]
        # list of reference streams, one per reference
        ref_streams = [[r] for r in references]
        # BLEU against references
        bleu_ref = self.metric.corpus_score(sys_stream, ref_streams).score

        # BLEU against source (self-BLEU)
        src_streams = [[input]]
        bleu_self = self.metric.corpus_score(sys_stream, src_streams).score

        return alpha_val * bleu_ref - (1 - alpha_val) * bleu_self 