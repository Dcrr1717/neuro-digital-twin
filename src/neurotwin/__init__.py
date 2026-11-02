"""neurotwin: a fast, uncertainty-aware deep-learning twin of a Hodgkin-Huxley neuron."""

from .hh import PARAM_NAMES, HHParams, simulate_rk4_batch, simulate_scipy
from .models import Ensemble, GRUSurrogate, NeuralODESurrogate, Normalizer

__all__ = [
    "PARAM_NAMES",
    "Ensemble",
    "GRUSurrogate",
    "HHParams",
    "NeuralODESurrogate",
    "Normalizer",
    "simulate_rk4_batch",
    "simulate_scipy",
]

__version__ = "0.1.0"
