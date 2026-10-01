"""节点级结构不确定度 u_i / 可信度 c_i（与 MVGAE 式 131–133 对齐）。"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from model.PYG_MVGAE_model import MVGAE


def node_u_and_c_from_logstd(logstd: Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """u_i = mean_r(σ_{i,r}), c_i = exp(-u_i)。"""
    u = MVGAE.node_uncertainty(logstd).detach().cpu().numpy().astype(np.float64)
    c = MVGAE.structure_confidence(torch.as_tensor(u)).numpy().astype(np.float64)
    return u, c


def node_u_and_c_from_sigma(sigma: Tensor) -> Tuple[np.ndarray, np.ndarray]:
    u = sigma.mean(dim=-1).detach().cpu().numpy().astype(np.float64)
    c = np.exp(-np.clip(u, 0.0, None))
    return u, c


def phy_degree_features(a_phy: np.ndarray) -> Dict[str, np.ndarray]:
    """出度/入度/总度（有向物理图）。"""
    a = (np.asarray(a_phy) > 0).astype(np.float64)
    np.fill_diagonal(a, 0.0)
    out_deg = a.sum(axis=1)
    in_deg = a.sum(axis=0)
    return {
        "out_degree": out_deg,
        "in_degree": in_deg,
        "degree": out_deg + in_deg,
    }


def summarize_node_uncertainty(
    u: np.ndarray,
    c: np.ndarray,
    a_phy: np.ndarray,
    coords: Optional[np.ndarray] = None,
    node_mae: Optional[np.ndarray] = None,
    p_low: float = 33.0,
    p_high: float = 66.0,
) -> Dict[str, Any]:
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    c = np.asarray(c, dtype=np.float64).reshape(-1)
    n = u.shape[0]
    degs = phy_degree_features(a_phy)
    deg = degs["degree"]

    p33 = float(np.percentile(u, p_low))
    p66 = float(np.percentile(u, p_high))
    tier = np.zeros(n, dtype=np.int8)
    tier[u < p33] = 1
    tier[(u >= p33) & (u < p66)] = 2
    tier[u >= p66] = 3

    def _corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
        if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    tiers = {}
    for name, t in (("low", 1), ("medium", 2), ("high", 3)):
        m = tier == t
        tiers[name] = {
            "n": int(m.sum()),
            "ratio": float(m.mean()),
            "mean_u": float(u[m].mean()) if m.any() else None,
            "mean_c": float(c[m].mean()) if m.any() else None,
            "mean_degree": float(deg[m].mean()) if m.any() else None,
            "mean_out_degree": float(degs["out_degree"][m].mean()) if m.any() else None,
            "mean_in_degree": float(degs["in_degree"][m].mean()) if m.any() else None,
        }
        if node_mae is not None and m.any():
            tiers[name]["mean_node_mae"] = float(np.asarray(node_mae)[m].mean())

    summary: Dict[str, Any] = {
        "n_nodes": n,
        "u": {
            "mean": float(u.mean()),
            "std": float(u.std()),
            "min": float(u.min()),
            "max": float(u.max()),
            "p33": p33,
            "p66": p66,
            "p66_minus_p33": float(p66 - p33),
        },
        "c": {
            "mean": float(c.mean()),
            "std": float(c.std()),
            "min": float(c.min()),
            "max": float(c.max()),
        },
        "corr": {
            "u_vs_degree": _corr(u, deg),
            "u_vs_out_degree": _corr(u, degs["out_degree"]),
            "u_vs_in_degree": _corr(u, degs["in_degree"]),
            "c_vs_degree": _corr(c, deg),
        },
        "tiers_by_u": tiers,
    }
    if node_mae is not None:
        mae = np.asarray(node_mae, dtype=np.float64).reshape(-1)
        summary["corr"]["u_vs_node_mae"] = _corr(u, mae)
        summary["corr"]["c_vs_node_mae"] = _corr(c, mae)
        summary["node_mae"] = {
            "mean": float(mae.mean()),
            "std": float(mae.std()),
        }

    if coords is not None:
        xy = np.asarray(coords, dtype=np.float64)
        # 到质心的距离：边缘节点是否更不确定
        center = xy.mean(axis=0, keepdims=True)
        dist = np.linalg.norm(xy - center, axis=1)
        summary["corr"]["u_vs_dist_to_centroid"] = _corr(u, dist)
        for name, t in (("low", 1), ("medium", 2), ("high", 3)):
            m = tier == t
            if m.any():
                summary["tiers_by_u"][name]["mean_dist_to_centroid"] = float(dist[m].mean())

    return {"summary": summary, "tier": tier, "degree": deg}


def load_node_mae_from_pred_npz(path: str) -> np.ndarray:
    """
    读取 predict_and_save_results 保存的 npz：
      prediction, data_target_tensor 形状约为 [B, N, T] 或 [B, N, T, 1]
    返回每个节点在测试集上的 MAE（对 batch×时间平均；target==0 视为缺失并 mask）。
    """
    data = np.load(path)
    if "prediction" not in data or "data_target_tensor" not in data:
        raise KeyError("npz 需含 prediction 与 data_target_tensor: %s" % path)
    pred = np.asarray(data["prediction"], dtype=np.float64)
    tgt = np.asarray(data["data_target_tensor"], dtype=np.float64)
    while pred.ndim > 3:
        pred = pred.squeeze(-1)
    while tgt.ndim > 3:
        tgt = tgt.squeeze(-1)
    # [B,N,T]
    if pred.shape != tgt.shape:
        raise ValueError("prediction/target 形状不一致: %s vs %s" % (pred.shape, tgt.shape))
    err = np.abs(pred - tgt)
    mask = tgt != 0.0
    # 节点维 = 1
    num = (err * mask).sum(axis=(0, 2))
    den = mask.sum(axis=(0, 2)).clip(min=1.0)
    return (num / den).astype(np.float64)
