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
from autometrics.metrics.reference_free.UniEvalFact import UniEvalFact
from autometrics.metrics.reference_free.Perplexity import Perplexity

from autometrics.metrics.reference_free.FKGL import FKGL

reference_based_metrics = [BLEU(), CHRF(), TER(), GLEU(), SARI(), BERTScore(), ROUGE(), MOVERScore(), BARTScore(), UniEvalDialogue(), UniEvalSum()]
reference_free_metrics = [FKGL(), UniEvalFact(), Perplexity()]

all_metrics = reference_based_metrics + reference_free_metrics