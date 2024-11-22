import textstat
from autometrics.metrics.reference_free.ReferenceFreeMetric import ReferenceFreeMetric

# class FKGL():

#     name = "FKGL"

#     def compute_metric(self, complex, simplified, references):

#         all_scores = []
#         for simp in simplified:
#             score = textstat.flesch_kincaid_grade(simp)
#             all_scores.append(score)
#         return all_scores

class FKGL(ReferenceFreeMetric):

    def __init__(self, name="FKGL", description="Flesch-Kincaid Grade Level (FKGL) is a metric that estimates the readability of a text based on the average number of syllables per word and the average number of words per sentence. Lower scores indicate easier-to-read text."):
        super().__init__(name, description)

    def calculate(self, input, output, references=None, **kwargs):
        """
        Calculate the metric
        """
        return textstat.flesch_kincaid_grade(output)