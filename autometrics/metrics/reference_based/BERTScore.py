import bert_score
from autometrics.metrics.reference_based.ReferenceBasedMultiMetric import ReferenceBasedMultiMetric
from autometrics.metrics.reference_based.ReferenceBasedMetric import ReferenceBasedMetric

def compute_bertscore(original, output, references, model="roberta-large", type="all", compare_to_original=False):

    all_origs, all_refs, all_cands = [], [], []
    for orig, hyp, refs in zip(original, output, references):
        for ref in refs:
            all_refs.append(ref.lower())
            all_cands.append(hyp.lower())
            all_origs.append(orig.lower())

    if compare_to_original:
        (P, R, F), _ = bert_score.score(all_cands, all_origs, lang="en", return_hash=True, verbose=True, idf=False,
                                        model_type=model)
    else:
        (P, R, F), _ = bert_score.score(all_cands, all_refs, lang="en", return_hash=True, verbose=True, idf=False,
                                    model_type=model)

    ind = 0
    pscores = []
    rscores = []
    fscores = []
    for orig, out, refs in zip(original, output, references):
        sub_pscores = []
        sub_rscores = []
        sub_fscores = []
        for _ in refs:
            sub_fscores.append(F[ind].item())
            sub_pscores.append(P[ind].item())
            sub_rscores.append(R[ind].item())
            ind += 1
        pscores.append(max(sub_pscores))
        rscores.append(max(sub_rscores))
        fscores.append(max(sub_fscores))

    assert len(pscores) == len(rscores) == len(fscores) == len(output) == len(references) == len(original)
    
    if type == "precision":
        return pscores
    elif type == "recall":
        return rscores
    elif type == "f1":
        return fscores
    elif type == "all":
        return pscores, rscores, fscores

class BERTScore(ReferenceBasedMultiMetric):

    def __init__(self, model="roberta-large"):
        name = "BERTScore_" + model
        description = "BERTScore is a metric that computes the similarity between two sentences using a pre-trained BERT model. It is based on the cosine similarity between the embeddings of the two sentences."
        self.model = model

        submetrics = ["P", "R", "F"]
        
        super().__init__(name, description, [f"BERTScore{submetric}_{model}" for submetric in submetrics])
        
    def calculate_batched(self, inputs, outputs, references=None, **kwargs):
        """
        Calculate the BERTScore for a batch of inputs and outputs.
        """
        if references is None:
            references = [None] * len(inputs)

        return compute_bertscore(inputs, outputs, references, model=self.model, type="all")
    
    def calculate(self, input, output, references=None, **kwargs):
        """
        Calculate the BERTScore for a single input/output pair.
        """
        if references is None:
            references = []

        p,r,f = compute_bertscore([input], [output], [references], model=self.model, type="all")
        return p[0], r[0], f[0]

