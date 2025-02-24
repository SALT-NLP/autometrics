from autometrics.metrics.reference_based.ReferenceBasedMultiMetric import ReferenceBasedMultiMetric

from autometrics.metrics.unieval.utils import convert_to_json
from autometrics.metrics.unieval.evaluator import get_evaluator

import torch

class UniEvalDialogue(ReferenceBasedMultiMetric):
    """"""

    def __init__(self):
        name = "UniEvalDialogue"
        description = "UniEvalDialogue is a metric for evaluating the quality of generated dialogues. It uses a pre-trained model to assess various dimensions of the dialogue, such as naturalness, coherence, engagingness, groundedness, and understandability. The metric provides a score based on the model's predictions, allowing for a quantitative evaluation of the dialogue's quality."
        self.submetrics = ["naturalness", "coherence", "engagingness", "groundedness", "understandability"]

        self.task = 'dialogue'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.evaluator = get_evaluator(self.task, device=self.device)
        
        super().__init__(name, description, ["UniEvalDialogue-" + submetric for submetric in self.submetrics])

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

        # Prepare data for pre-trained v
        data = convert_to_json(output_list=[output], src_list=[input], context_list=references)

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
        data = convert_to_json(output_list=outputs, src_list=inputs, context_list=[reference[0] for reference in references])

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
    unieval = UniEvalDialogue()
    input = "hi , do you know much about the internet ? \n i know a lot about different sites and some website design , how about you ? \n\n"
    output = "i do too . did you know the 3 horizontal line menu on apps and websites is called the hamburger button ?"
    references = ["the 3 horizontal line menu on apps and websites is called a hamburger button .\n"]
    scores = unieval.calculate(input, output, references)
    print("UniEvalDialogue scores:", scores)
    # Test batch processing
    inputs = [input, "hey, what do you think about the weather today?"]
    outputs = [output, "it's sunny and warm outside."]
    references = [references, ["the weather is nice today."]]
    scores = unieval.calculate_batched(inputs, outputs, references)
    print("UniEvalDialogue batch scores:", scores)
