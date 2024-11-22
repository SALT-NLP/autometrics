from autometrics.metrics.reference_based.BLEU import BLEU
from autometrics.metrics.reference_based.CHRF import CHRF
from autometrics.metrics.reference_based.TER import TER
from autometrics.metrics.reference_based.GLEU import GLEU
from autometrics.metrics.reference_based.SARI import SARI
from autometrics.metrics.reference_based.BERTScore import BERTScore

from autometrics.metrics.reference_free.FKGL import FKGL

reference_based_metrics = [BLEU(), CHRF(), TER(), GLEU(), SARI(), BERTScore()]
reference_free_metrics = [FKGL()]

all_metrics = reference_based_metrics + reference_free_metrics