"""
长期结构先验（式 146–172）：

  A^{latent} → Top-K_p + τ_p → B^{latent}
  A^{base} = η A^{phy} + (1-η) B^{latent}，η = sigmoid(θ_η)
  A^{prior} = D_c A^{base} D_c  （c_i c_j 逐边校准）
  Â^{prior} = (D^{prior})^{-1}(A^{prior}+I)  行归一化长期先验
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

EPS = 1e-8


def sparsify_latent_adjacency(
    a_latent: Union[Tensor, np.ndarray],
    top_k: int = 10,
    tau_p: float = 0.1,
) -> Union[Tensor, np.ndarray]:
    """
    式(148)(149)：逐源节点行 Top-K_p，且 A_ij >= τ_p；对角清零；不对称化。
    """
    is_numpy = isinstance(a_latent, np.ndarray)
    if is_numpy:
        a = torch.from_numpy(np.asarray(a_latent, dtype=np.float32))
    else:
        a = a_latent.float()

    n = a.size(0)
    if a.size(0) != a.size(1):
        raise ValueError("A^{latent} 须为方阵")

    k = max(1, min(int(top_k), n - 1))
    tau = float(tau_p)

    scores = a.clone()
    scores.fill_diagonal_(-float("inf"))
    topv, topi = torch.topk(scores, k=k, dim=1)

    b = torch.zeros_like(a)
    keep = topv >= tau
    rows = torch.arange(n, device=a.device).unsqueeze(1).expand_as(topi)
    b[rows[keep], topi[keep]] = a[rows[keep], topi[keep]]
    b.fill_diagonal_(0.0)

    if is_numpy:
        return b.detach().cpu().numpy().astype(np.float32)
    return b


def fuse_base_adjacency(
    a_phy: Union[Tensor, np.ndarray],
    b_latent: Union[Tensor, np.ndarray],
    eta: float,
) -> Union[Tensor, np.ndarray]:
    """式(151)：A^{base} = η A^{phy} + (1-η) B^{latent}。"""
    is_numpy = isinstance(a_phy, np.ndarray)
    if is_numpy:
        ap = np.asarray(a_phy, dtype=np.float32)
        bl = np.asarray(b_latent, dtype=np.float32)
        e = float(np.clip(eta, 0.0, 1.0))
        out = e * ap + (1.0 - e) * bl
        np.fill_diagonal(out, 0.0)
        return out.astype(np.float32)

    out = float(eta) * a_phy.float() + (1.0 - float(eta)) * b_latent.float()
    out = out.clone()
    out.fill_diagonal_(0.0)
    return out


def calibrate_with_confidence(
    a_base: Tensor,
    c_v: Tensor,
) -> Tensor:
    """
    式(162)(166)：A^{prior_raw} = D_c A^{base} D_c = A^{base} ⊙ (c c^T)。
    A_ij ← c_i * A_ij * c_j；对角保持 0。
    """
    c = c_v.reshape(-1).float().clamp(EPS, 1.0)
    a = a_base.float()
    # (c c^T)_ij = c_i c_j
    a_cal = a * c.unsqueeze(1) * c.unsqueeze(0)
    a_cal = a_cal.clone()
    a_cal.fill_diagonal_(0.0)
    return a_cal


def normalize_structure_prior(a_calibrated: Tensor, eps: float = EPS) -> Tensor:
    """
    式(170)–(172)：Â = D^{-1}(A + I)。
    """
    n = a_calibrated.size(0)
    a = a_calibrated.float().clone()
    eye = torch.eye(n, device=a.device, dtype=a.dtype)
    a = a + eye
    deg = a.sum(dim=1).clamp_min(eps)
    return a / deg.unsqueeze(1)


def asym_normalize_torch(adj: Tensor, eps: float = EPS) -> Tensor:
    """行归一化 Â = D^{-1} A（有向）。"""
    deg = adj.sum(dim=1).clamp_min(eps)
    return adj / deg.unsqueeze(1)


def build_supports_from_aprior(
    a_prior: Tensor,
    device: Optional[torch.device] = None,
    already_normalized: bool = True,
) -> list[Tensor]:
    """
    由 A^{prior} 构造 doubletransition supports。
    already_normalized=True：A^{prior} 已是式(172)结果，不再重复行归一化第一项。
    """
    if device is not None:
        a_prior = a_prior.to(device)
    if already_normalized:
        s0 = a_prior
        s1 = asym_normalize_torch(a_prior.T)
    else:
        s0 = asym_normalize_torch(a_prior)
        s1 = asym_normalize_torch(a_prior.T)
    return [s0, s1]


# 兼容旧名
build_supports_from_abase = build_supports_from_aprior


class StructureBasePrior(nn.Module):
    """
    不确定性校准长期结构先验（式 151–172）：
      A^{base} = η A^{phy} + (1-η) B^{latent}
      A^{cal}  = D_c A^{base} D_c
      A^{prior}= D^{-1}(A^{cal}+I)
    并缓存 A_0^{prior} 供式(178) 结构保持损失使用。
    """

    def __init__(
        self,
        a_phy: Union[Tensor, np.ndarray],
        a_latent: Optional[Union[Tensor, np.ndarray]] = None,
        b_latent: Optional[Union[Tensor, np.ndarray]] = None,
        c_v: Optional[Union[Tensor, np.ndarray]] = None,
        top_k: int = 10,
        tau_p: float = 0.1,
        theta_eta_init: float = 0.0,
    ):
        super().__init__()
        if a_latent is None and b_latent is None:
            raise ValueError("需要提供 a_latent 或 b_latent")

        ap = torch.as_tensor(a_phy, dtype=torch.float32)
        if b_latent is None:
            al = torch.as_tensor(a_latent, dtype=torch.float32)
            bl = sparsify_latent_adjacency(al, top_k=top_k, tau_p=tau_p)
        else:
            bl = torch.as_tensor(b_latent, dtype=torch.float32)

        if ap.shape != bl.shape:
            raise ValueError(
                "A^{phy} 与 B^{latent} 形状不一致: %s vs %s"
                % (tuple(ap.shape), tuple(bl.shape))
            )

        n = ap.size(0)
        if c_v is None:
            cv = torch.ones(n, dtype=torch.float32)
        else:
            cv = torch.as_tensor(c_v, dtype=torch.float32).reshape(-1)
            if cv.numel() != n:
                raise ValueError("C_v 长度 %d 与节点数 %d 不一致" % (cv.numel(), n))
            cv = cv.clamp(EPS, 1.0)

        self.register_buffer("a_phy", ap)
        self.register_buffer("b_latent", bl)
        self.register_buffer("c_v", cv)
        self.top_k = int(top_k)
        self.tau_p = float(tau_p)
        self.theta_eta = nn.Parameter(torch.tensor(float(theta_eta_init)))

        # 式(176)：冻结初始先验锚点 A_0^{prior}
        with torch.no_grad():
            a0 = self._normalized_prior(self._base_adjacency(self.eta.detach()))
        self.register_buffer("a0_prior", a0)
        self._frozen_to_a0 = False

    def set_frozen_to_a0(self, flag: bool) -> None:
        """第二阶段：A^{prior} := A_0^{prior}（式 296）。"""
        self._frozen_to_a0 = bool(flag)

    @property
    def eta(self) -> Tensor:
        return torch.sigmoid(self.theta_eta)

    def _base_adjacency(self, eta: Tensor) -> Tensor:
        a_base = eta * self.a_phy + (1.0 - eta) * self.b_latent
        a_base = a_base.clone()
        a_base.fill_diagonal_(0.0)
        return a_base

    def base_adjacency(self) -> Tensor:
        """式(151)：当前 A^{base}。"""
        return self._base_adjacency(self.eta)

    def calibrated_adjacency(self) -> Tensor:
        """式(162)：D_c A^{base} D_c（未加自环、未行归一化）。"""
        return calibrate_with_confidence(self.base_adjacency(), self.c_v)

    def _normalized_prior(self, a_base: Tensor) -> Tensor:
        a_cal = calibrate_with_confidence(a_base, self.c_v)
        return normalize_structure_prior(a_cal)

    def forward(self) -> Tensor:
        """返回当前归一化 A^{prior}（式 172）；冻结阶段返回 A_0^{prior}。"""
        if getattr(self, "_frozen_to_a0", False):
            return self.a0_prior
        return self._normalized_prior(self.base_adjacency())

    def prior_preservation_loss(self, eps: float = 1e-8) -> Tensor:
        """
        式(177)(178)：
          Ω_p = Supp(A_0^{prior}) ∪ Supp(A^{prior})
          L_prior = mean_{(i,j)∈Ω_p} (A_ij^{prior} - StopGrad(A0_ij^{prior}))^2
        """
        a_cur = self.forward()
        a0 = self.a0_prior.detach()  # StopGrad
        mask = (a_cur > eps) | (a0 > eps)
        if not mask.any():
            return a_cur.new_zeros(())
        diff = (a_cur - a0)[mask]
        return diff.pow(2).mean()

    def edge_counts(self) -> dict:
        with torch.no_grad():
            a_base = self.base_adjacency()
            a_cal = calibrate_with_confidence(a_base, self.c_v)
            a_prior = self.forward()
            return {
                "phy_nnz": int((self.a_phy > 0).sum().item()),
                "b_latent_nnz": int((self.b_latent > 0).sum().item()),
                "a_base_nnz": int((a_base > 0).sum().item()),
                "a_cal_nnz": int((a_cal > 0).sum().item()),
                "a_prior_nnz": int((a_prior > 1e-8).sum().item()),
                "eta": float(self.eta.item()),
                "c_mean": float(self.c_v.mean().item()),
            }
