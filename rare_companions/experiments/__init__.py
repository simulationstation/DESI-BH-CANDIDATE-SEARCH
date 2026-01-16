"""
Experiment modules for rare companion searches.
"""

from .base import BaseExperiment, ExperimentResult, Candidate
from .e1_mass_gap import MassGapExperiment
from .e2_dark_companions import DarkCompanionExperiment
from .e3_dwd_lisa import DWDLISAExperiment
from .e4_brown_dwarf import BrownDwarfExperiment
from .e5_hierarchical import HierarchicalExperiment
from .e6_accretion import AccretionExperiment
from .e7_halo_cluster import HaloClusterExperiment
from .e8_anomalies import AnomalyExperiment

EXPERIMENTS = {
    'E1_mass_gap': MassGapExperiment,
    'E2_dark_companions': DarkCompanionExperiment,
    'E3_dwd_lisa': DWDLISAExperiment,
    'E4_brown_dwarf': BrownDwarfExperiment,
    'E5_hierarchical': HierarchicalExperiment,
    'E6_accretion': AccretionExperiment,
    'E7_halo_cluster': HaloClusterExperiment,
    'E8_anomalies': AnomalyExperiment,
}
