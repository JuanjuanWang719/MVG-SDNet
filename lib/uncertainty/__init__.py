"""不确定性分析工具包（独立于训练主流程，供 tools/run_uncertainty_*.py 调用）。"""

from lib.uncertainty.edge_variance import (
    EdgeUncertaintyResult,
    build_candidate_edge_mask,
    compute_edge_uncertainty_from_samples,
    summarize_uncertainty_tiers,
)
from lib.uncertainty.node_uncertainty import (
    node_u_and_c_from_sigma,
    summarize_node_uncertainty,
)
from lib.uncertainty.posterior import load_mvgae_posterior

__all__ = [
    "EdgeUncertaintyResult",
    "build_candidate_edge_mask",
    "compute_edge_uncertainty_from_samples",
    "summarize_uncertainty_tiers",
    "node_u_and_c_from_sigma",
    "summarize_node_uncertainty",
    "load_mvgae_posterior",
]
