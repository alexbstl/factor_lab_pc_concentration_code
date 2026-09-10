"""Factor Lab Manifold Analysis - Paper 1 subset"""
__version__ = "3.0.0"

# Core types and functions
from .factor_types import FactorModelData

# Distributions
from .distributions import create_sampler

# Flexible simulation components
from .model_builder import FactorModelBuilder
from .flexible_simulator import ReturnsSimulator

# Analysis framework
from .analysis import SimulationContext
from .analyses import (
    Analyses,
    ManifoldDistanceAnalysis,
    ImplicitEigenAnalysis,
    EigenvectorAlignment,
)

__all__ = [
    'FactorModelData',
    'create_sampler',
    'FactorModelBuilder',
    'ReturnsSimulator',
    'SimulationContext',
    'Analyses',
    'ManifoldDistanceAnalysis',
    'ImplicitEigenAnalysis',
    'EigenvectorAlignment',
]
