from autometrics.metrics.reference_based.BLEU import BLEU
from autometrics.metrics.reference_based.CHRF import CHRF
from autometrics.metrics.reference_based.TER import TER
from autometrics.metrics.reference_based.GLEU import GLEU
from autometrics.metrics.reference_based.SARI import SARI
from autometrics.metrics.reference_based.BERTScore import BERTScore
from autometrics.metrics.reference_based.ROUGE import ROUGE
from autometrics.metrics.reference_based.MOVERScore import MOVERScore
from autometrics.metrics.reference_based.BARTScore import BARTScore
from autometrics.metrics.reference_based.UniEvalDialogue import UniEvalDialogue
from autometrics.metrics.reference_based.UniEvalSum import UniEvalSum
from autometrics.metrics.reference_based.CIDEr import CIDEr
# from autometrics.metrics.reference_based.METEOR import METEOR
# from autometrics.metrics.reference_based.QUAC import QUAC
from autometrics.metrics.reference_based.StringSimilarity import LevenshteinDistance
from autometrics.metrics.reference_based.StringSimilarity import LevenshteinRatio
# from autometrics.metrics.reference_based.StringSimilarity import JaccardSimilarity
from autometrics.metrics.reference_based.StringSimilarity import HammingDistance
from autometrics.metrics.reference_based.StringSimilarity import JaroSimilarity
from autometrics.metrics.reference_based.StringSimilarity import JaroWinklerSimilarity
from autometrics.metrics.reference_based.ParaScore import ParaScore
from autometrics.metrics.reference_based.YiSi import YiSi

from autometrics.metrics.reference_free.FKGL import FKGL
from autometrics.metrics.reference_free.UniEvalFact import UniEvalFact
from autometrics.metrics.reference_free.Perplexity import Perplexity
from autometrics.metrics.reference_free.ParaScoreFree import ParaScoreFree
from autometrics.metrics.reference_free.INFORMRewardModel import INFORMRewardModel
from autometrics.metrics.reference_free.PRMRewardModel import MathProcessRewardModel

reference_based_metrics = [BLEU(), CHRF(), TER(), GLEU(), SARI(), BERTScore(), ROUGE(), MOVERScore(), BARTScore(), UniEvalDialogue(), UniEvalSum(), CIDEr(), LevenshteinDistance(), LevenshteinRatio(), HammingDistance(), JaroSimilarity(), JaroWinklerSimilarity(), ParaScore(), YiSi()]
reference_free_metrics = [FKGL(), UniEvalFact(), Perplexity(batch_size=2), ParaScoreFree(), INFORMRewardModel(), MathProcessRewardModel()]
                          
all_metrics = reference_based_metrics + reference_free_metrics