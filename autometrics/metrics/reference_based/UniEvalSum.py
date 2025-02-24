from autometrics.metrics.reference_based.ReferenceBasedMultiMetric import ReferenceBasedMultiMetric

from autometrics.metrics.unieval.utils import convert_to_json
from autometrics.metrics.unieval.evaluator import get_evaluator

import torch

class UniEvalSum(ReferenceBasedMultiMetric):
    """"""

    def __init__(self):
        name = "UniEvalSum"
        description = "UniEvalSum is a metric for evaluating the quality of generated summaries. It uses a pre-trained model to assess various dimensions of the summary, such as fluency, coherence, and relevance. The metric provides a score based on the model's predictions, allowing for a quantitative evaluation of the summary's quality."
        self.submetrics = ["fluency", "coherence", "consistency", "relevance"]

        self.task = 'summarization'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.evaluator = get_evaluator(self.task, device=self.device)
        
        super().__init__(name, description, ["UniEvalSum-" + submetric for submetric in self.submetrics])

    def _parse_unieval(self, result):
      results = [result[submetric] for submetric in self.submetrics]
      return results
    
    def calculate(self, input, output, references=None, **kwargs):
        """
        Calculate UniEvalSum scores for the given input and output.
        """
        if references is None:
            references = []

        if len(references) > 1:
            references = [references[0]]

        # Prepare data for pre-trained evaluators
        data = convert_to_json(output_list=[output], src_list=[input], ref_list=references)

        # Get multi-dimensional evaluation scores
        eval_scores = self.evaluator.evaluate(data)
        
        return self._parse_unieval(eval_scores[0])
    
    def calculate_batched(self, inputs, outputs, references=None, **kwargs):
        """
        Calculate UniEvalSum scores for the given inputs and outputs in batches.
        """
        if references is None:
            references = [[] for _ in range(len(inputs))]

        # Prepare data for pre-trained evaluators
        data = convert_to_json(output_list=outputs, src_list=inputs, ref_list=[reference[0] for reference in references])

        # Get multi-dimensional evaluation scores
        eval_scores = self.evaluator.evaluate(data)
        
        results = [self._parse_unieval(eval_score) for eval_score in eval_scores]

        # unzip the results
        results = list(zip(*results))

        # Convert to list of lists
        results = [list(result) for result in results]

        return results
    
if __name__ == "__main__":
    # Example usage
    unieval = UniEvalSum()
    input = "Peter and Elizabeth took a taxi to attend the night party in the city. \
             While in the party, Elizabeth collapsed and was rushed to the hospital."
    output = "Peter and Elizabeth attend party city. Elizabeth rushed hospital."
    references = ["Elizabeth was hospitalized after attending a party with Peter."]
    scores = unieval.calculate(input, output, references)
    print("UniEvalSum scores:", scores)

    # Test batch processing
    inputs = [
        "Peter and Elizabeth took a taxi to attend the night party in the city. \
             While in the party, Elizabeth collapsed and was rushed to the hospital.",
        "The cat sat on the mat."
    ]
    outputs = [
        "Peter and Elizabeth attend party city. Elizabeth rushed hospital.",
        "The cat is on the mat."
    ]
    references = [
        ["Elizabeth was hospitalized after attending a party with Peter."],
        ["The cat sat on the mat."]
    ]
    scores = unieval.calculate_batched(inputs, outputs, references)
    print("UniEvalSum batch scores:", scores)
