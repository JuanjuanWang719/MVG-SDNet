"""
实验 1 / 1b：边关系不确定性
  z_i^(m) = μ_i + σ_i ⊙ ε^(m)
  Exp1 : A_ij^(m) = sigmoid(s_ij^(m)),  U = Var(A)
  Exp1b: s_ij^(m) = 非对称双线性 score（未 sigmoid）, U = Var(s)
  总体方差 1/M；可按全体边或候选边集（phy ∪ Top-K ∪ 负边）分层
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from lib.uncertainty.posterior import sample_latent_z
from model.PYG_MVGAE_model import AsymmetricBilinearDecoder


@dataclass
class EdgeUncertaintyResult:
    u: np.ndarray  # [N,N]
    a_mean: np.ndarray  # [N,N] 或 score 均值（视 mode）
    p33: float
    p66: float
    tier: np.ndarray  # [N,N] int8: 0 未评估, 1 low, 2 mid, 3 high
    summary: Dict[str, Any]


def decode_dense_score(model, z: Tensor) -> Tensor:
    """未 sigmoid 的稠密 score s_ij（对角清零）。"""
    dec = model.decoder
    if isinstance(dec, AsymmetricBilinearDecoder):
        zs = dec.W_src(z)
        zd = dec.W_dst(z)
        s = (zs @ zd.T) / float(dec.latent_dim)
        s = s.clamp(-20.0, 20.0)
        s.fill_diagonal_(0.0)
        return s
    s = z @ z.T
    s.fill_diagonal_(0.0)
    return s


def decode_dense_adjacency(model, z: Tensor) -> Tensor:
    """与预训练一致的稠密 A^{latent}。"""
    dec = model.decoder
    if isinstance(dec, AsymmetricBilinearDecoder):
        return dec.dense(z)
    adj = torch.sigmoid(z @ z.T)
    adj.fill_diagonal_(0.0)
    return adj


@torch.no_grad()
def compute_edge_uncertainty_from_samples(
    model,
    mu: Tensor,
    sigma: Tensor,
    num_samples: int = 50,
    seed: int = 42,
    mode: str = "prob",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    流式累计，避免存满 [M,N,N]。
    mode=prob  → U=Var(A), 同时返回 E[A]
    mode=score → U=Var(s), 同时返回 E[s]
    总体方差：E[x^2]-(E[x])^2
    """
    m = int(num_samples)
    if m < 2:
        raise ValueError("num_samples 至少为 2")
    mode = str(mode).strip().lower()
    if mode not in ("prob", "score", "a", "logit", "s"):
        raise ValueError("mode 应为 prob 或 score，得到: %r" % mode)
    use_score = mode in ("score", "logit", "s")

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    sum_x: Optional[Tensor] = None
    sum_x2: Optional[Tensor] = None
    for _ in range(m):
        z = sample_latent_z(mu, sigma, generator=g)
        x = decode_dense_score(model, z) if use_score else decode_dense_adjacency(model, z)
        if sum_x is None:
            sum_x = x.clone()
            sum_x2 = x * x
        else:
            sum_x = sum_x + x
            sum_x2 = sum_x2 + x * x
    assert sum_x is not None and sum_x2 is not None
    mean = sum_x / float(m)
    var = (sum_x2 / float(m) - mean * mean).clamp_min(0.0)
    var.fill_diagonal_(0.0)
    mean.fill_diagonal_(0.0)
    return var.detach().cpu().numpy().astype(np.float64), mean.detach().cpu().numpy().astype(
        np.float32
    )


def build_candidate_edge_mask(
    a_phy: np.ndarray,
    score_or_prob_ref: np.ndarray,
    top_k_per_node: int = 15,
    num_neg_per_phy: int = 1,
    seed: int = 42,
) -> np.ndarray:
    """
    候选有向边 = 物理边 ∪ 每行 Top-K(ref) ∪ 与物理边等量的随机负边。
    ref 建议用 μ 下的 A 或 score。
    """
    n = a_phy.shape[0]
    eye = np.eye(n, dtype=bool)
    phy = (a_phy > 0) & (~eye)
    mask = phy.copy()

    ref = np.asarray(score_or_prob_ref, dtype=np.float64).copy()
    np.fill_diagonal(ref, -np.inf)
    k = max(1, min(int(top_k_per_node), n - 1))
    parti = np.argpartition(-ref, kth=k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    mask[rows, parti.reshape(-1)] = True
    mask[eye] = False

    rng = np.random.RandomState(int(seed))
    n_phy = int(phy.sum())
    n_neg = max(0, int(num_neg_per_phy) * n_phy)
    if n_neg > 0:
        candidates = np.argwhere(~mask & ~eye)
        if candidates.shape[0] > 0:
            take = min(n_neg, candidates.shape[0])
            pick = rng.choice(candidates.shape[0], size=take, replace=False)
            ij = candidates[pick]
            mask[ij[:, 0], ij[:, 1]] = True

    return mask


def summarize_uncertainty_tiers(
    u: np.ndarray,
    a_phy: np.ndarray,
    p_low: float = 33.0,
    p_high: float = 66.0,
    eval_mask: Optional[np.ndarray] = None,
    value_mean: Optional[np.ndarray] = None,
    value_name: str = "A_mean",
) -> EdgeUncertaintyResult:
    """
    在 eval_mask 指定的有向边上算分位数并分层（默认全体 i≠j）。
    tier: 1=Low (<P33), 2=Medium, 3=High (>=P66)；mask 外为 0
    """
    n = u.shape[0]
    eye = np.eye(n, dtype=bool)
    if eval_mask is None:
        off = ~eye
    else:
        off = np.asarray(eval_mask, dtype=bool) & (~eye)

    vals = u[off]
    if vals.size == 0:
        raise ValueError("eval_mask 内无有效边")
    p33 = float(np.percentile(vals, p_low))
    p66 = float(np.percentile(vals, p_high))

    tier = np.zeros((n, n), dtype=np.int8)
    low = off & (u < p33)
    mid = off & (u >= p33) & (u < p66)
    high = off & (u >= p66)
    tier[low] = 1
    tier[mid] = 2
    tier[high] = 3

    phy = (a_phy > 0) & (~eye)
    phy_in = phy & off
    non_phy_in = (~phy) & off
    n_off = int(off.sum())
    n_phy = int(phy.sum())
    n_phy_in = int(phy_in.sum())

    def _count(mask: np.ndarray) -> int:
        return int(mask.sum())

    def _phy_frac_in(mask: np.ndarray) -> float:
        c = _count(mask)
        if c == 0:
            return 0.0
        return float((mask & phy).sum()) / float(c)

    def _tier_frac_in_phy(t: int) -> float:
        if n_phy_in == 0:
            return 0.0
        return float(((tier == t) & phy_in).sum()) / float(n_phy_in)

    summary: Dict[str, Any] = {
        "n_nodes": n,
        "n_eval_edges": n_off,
        "n_directed_pairs": int((~eye).sum()),
        "n_phy_edges": n_phy,
        "n_phy_in_eval": n_phy_in,
        "u_mean": float(vals.mean()),
        "u_std": float(vals.std()),
        "u_min": float(vals.min()),
        "u_max": float(vals.max()),
        "p33": p33,
        "p66": p66,
        "p66_minus_p33": float(p66 - p33),
        "tier_counts": {
            "low": _count(low),
            "medium": _count(mid),
            "high": _count(high),
        },
        "tier_ratios": {
            "low": _count(low) / float(n_off),
            "medium": _count(mid) / float(n_off),
            "high": _count(high) / float(n_off),
        },
        "phy_overlap_within_tier": {
            "low": _phy_frac_in(low),
            "medium": _phy_frac_in(mid),
            "high": _phy_frac_in(high),
        },
        "tier_fraction_within_phy_eval": {
            "low": _tier_frac_in_phy(1),
            "medium": _tier_frac_in_phy(2),
            "high": _tier_frac_in_phy(3),
        },
        "mean_u_on_phy": float(u[phy_in].mean()) if n_phy_in else None,
        "mean_u_on_non_phy": float(u[non_phy_in].mean()) if int(non_phy_in.sum()) else None,
    }
    if summary["mean_u_on_phy"] is not None and summary["mean_u_on_non_phy"] is not None:
        summary["mean_u_phy_minus_nonphy"] = float(
            summary["mean_u_on_phy"] - summary["mean_u_on_non_phy"]
        )

    if value_mean is not None:
        vm = np.asarray(value_mean)
        summary[value_name + "_on_eval"] = {
            "mean": float(vm[off].mean()),
            "std": float(vm[off].std()),
            "p10": float(np.percentile(vm[off], 10)),
            "p50": float(np.percentile(vm[off], 50)),
            "p90": float(np.percentile(vm[off], 90)),
        }
        by_tier: Dict[str, float] = {}
        for name, msk in (("low", low), ("medium", mid), ("high", high)):
            if msk.any():
                by_tier[name] = float(vm[msk].mean())
        summary[value_name + "_by_tier"] = by_tier

    return EdgeUncertaintyResult(
        u=u.astype(np.float32),
        a_mean=np.zeros_like(u, dtype=np.float32),
        p33=p33,
        p66=p66,
        tier=tier,
        summary=summary,
    )


def result_to_jsonable(result: EdgeUncertaintyResult, extra: Optional[Dict] = None) -> Dict:
    out = {
        "p33": result.p33,
        "p66": result.p66,
        "summary": result.summary,
    }
    if extra:
        out.update(extra)
    return out
