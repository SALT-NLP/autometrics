from autometrics.metrics.reference_based.BLEU import BLEU
from autometrics.metrics.reference_based.CHRF import CHRF
from autometrics.metrics.reference_based.TER import TER
from autometrics.metrics.reference_based.GLEU import GLEU

reference_based_metrics = [BLEU(), CHRF(), TER(), GLEU()]