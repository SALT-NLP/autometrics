from autometrics.metrics.reference_free.ReferenceFreeMultiMetric import ReferenceFreeMultiMetric

from autometrics.metrics.unieval.utils import convert_to_json
from autometrics.metrics.unieval.evaluator import get_evaluator

import torch

class UniEvalFact(ReferenceFreeMultiMetric):
    """"""

    def __init__(self):
        name = "UniEvalFact"
        description = "UniEvalFact is a metric for evaluating the factual consistency of generated text. It uses a pre-trained model to assess the factuality of the content, providing a score that indicates how well the generated text aligns with factual information. This metric is useful for tasks where factual accuracy is crucial, such as summarization and dialogue generation."
        self.submetrics = ["consistency"]

        self.task = 'fact'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.evaluator = get_evaluator(self.task, device=self.device)
        
        super().__init__(name, description, ["UniEvalFact-" + submetric for submetric in self.submetrics])

    def _parse_unieval(self, result):
      results = [result[submetric] for submetric in self.submetrics]
      return results
    
    def calculate(self, input, output, references=None, **kwargs):
        """
        Calculate UniEvalFact scores for the given input and output.
        """
        # Prepare data for pre-trained evaluators
        data = convert_to_json(output_list=[output], src_list=[input])

        # Get multi-dimensional evaluation scores
        eval_scores = self.evaluator.evaluate(data)
        
        return self._parse_unieval(eval_scores[0])
    
    def calculate_batched(self, inputs, outputs, references=None, **kwargs):
        """
        Calculate UniEvalFact scores for the given inputs and outputs in batches.
        """

        # Prepare data for pre-trained evaluators
        data = convert_to_json(output_list=outputs, src_list=inputs)

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
    unieval = UniEvalFact()
    input = "Peter and Elizabeth took a taxi to attend the night party in the city. \
             While in the party, Elizabeth collapsed and was rushed to the hospital."
    output = "Tom was rushed to hospital."

    scores = unieval.calculate(input, output)
    print("UniEvalFact scores:", scores)

    # Test batch processing
    inputs = [
        "Peter and Elizabeth took a taxi to attend the night party in the city. \
             While in the party, Elizabeth collapsed and was rushed to the hospital.",
        "Pancakes are a type of flatbread made from flour, water, and milk."
    ]
    outputs = [
        "Tom was rushed to hospital.",
        "Pancakes can be made with flour, water, and milk."
    ]
    scores = unieval.calculate_batched(inputs, outputs)
    print("UniEvalFact batch scores:", scores)

