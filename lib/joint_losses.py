"""
联合优化目标（式 273–287）：

  L_pred  : 等权多步 MAE
  L_tr    : 趋势二阶差分平滑
  L_prior : 长期结构保持（StructureBasePrior）
  L_pre   : 结构预训练目标（mask + λ_mv L_mv + β_KL L_KL）
  L_joint = L_pred + λ_tr L_tr + λ_prior L_prior + λ_str L_pre
  L_warm  = L_pred + λ_tr L_tr
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


def _valid_mask(target: Tensor, null_val: Optional[float]) -> Optional[Tensor]:
    """null_val=None → 不掩码；NaN → 掩掉 nan；其它 → 掩掉等于 null_val 的位置。"""
    if null_val is None:
        return None
    if null_val != null_val:  # NaN
        return ~torch.isnan(target)
    return target != float(null_val)


def prediction_mae_loss(
    pred: Tensor,
    target: Tensor,
    horizon_weights: Optional[Tensor] = None,
    null_val: Optional[float] = None,
) -> Tensor:
    """
    式(273)–(276)：多步 MAE。
    pred / target : [B, N, T'] 或 [B, N, C_y, T']
    horizon_weights : [T']，默认等权 1/T'
    null_val : 若给定，则忽略 target 中等于该值（或 NaN）的位置（METR/BAY 缺失常用 0）
    """
    if pred.shape != target.shape:
        raise ValueError("pred/target 形状不一致: %s vs %s" % (tuple(pred.shape), tuple(target.shape)))

    err = (pred - target).abs()
    mask = _valid_mask(target, null_val)
    if mask is not None:
        m = mask.to(dtype=err.dtype)
        if horizon_weights is None:
            denom = m.sum().clamp_min(1.0)
            return (err * m).sum() / denom
        w = horizon_weights.to(device=err.device, dtype=err.dtype)
        w = w / w.sum().clamp_min(1e-8)
        # 时间维在最后一维
        while w.dim() < err.dim():
            w = w.view(*([1] * (err.dim() - 1)), -1)
        denom = (m * w).sum().clamp_min(1e-8)
        return (err * m * w).sum() / denom

    if horizon_weights is None:
        return err.mean()

    w = horizon_weights.to(device=err.device, dtype=err.dtype)
    w = w / w.sum().clamp_min(1e-8)
    # 对非时间维求均值后，再对时间维加权
    if err.dim() == 3:
        # [B,N,T']
        per_h = err.mean(dim=(0, 1))
    elif err.dim() == 4:
        # [B,N,C,T']
        per_h = err.mean(dim=(0, 1, 2))
    else:
        raise ValueError("不支持的 pred 维数: %d" % err.dim())
    if per_h.numel() != w.numel():
        raise ValueError("horizon_weights 长度与 T' 不一致")
    return (per_h * w).sum()


def trend_second_order_smoothness_loss(h_tr: Tensor) -> Tensor:
    """
    式(278)(279)：
      Δ² h_τ = h_τ - 2 h_{τ-1} + h_{τ-2}
      L_tr = mean ||Δ² h||_2^2
    h_tr : [B, T, N, d_sp]
    """
    if h_tr.dim() != 4:
        raise ValueError("h_tr 须为 [B, T, N, d]")
    t = h_tr.size(1)
    if t < 3:
        return h_tr.new_zeros(())
    d2 = h_tr[:, 2:] - 2.0 * h_tr[:, 1:-1] + h_tr[:, :-2]
    return d2.pow(2).mean()


def kl_beta_warmup(epoch: int, beta_max: float, warmup_epochs: int) -> float:
    """式(283)：β_KL(e) = min(β_max, e/E_KL * β_max)。epoch 从 1 计。"""
    beta_max = float(beta_max)
    if warmup_epochs is None or int(warmup_epochs) <= 0:
        return beta_max
    e = max(1, int(epoch))
    e_kl = int(warmup_epochs)
    return float(min(beta_max, (e / float(e_kl)) * beta_max))


class JointForecastLoss(nn.Module):
    """
    组装 L_warm / L_joint。
    外部传入可选的 L_prior、L_pre 标量。
    """

    def __init__(
        self,
        lambda_tr: float = 0.01,
        lambda_prior: float = 0.1,
        lambda_str: float = 0.0,
        use_horizon_weights: bool = False,
        null_val: Optional[float] = None,
    ):
        super().__init__()
        self.lambda_tr = float(lambda_tr)
        self.lambda_prior = float(lambda_prior)
        self.lambda_str = float(lambda_str)
        self.use_horizon_weights = bool(use_horizon_weights)
        self.null_val = null_val
        self.last = {
            "pred": 0.0,
            "tr": 0.0,
            "prior": 0.0,
            "str": 0.0,
            "total": 0.0,
        }

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        h_tr: Optional[Tensor] = None,
        loss_prior: Optional[Tensor] = None,
        loss_str: Optional[Tensor] = None,
        warm_only: bool = False,
        horizon_weights: Optional[Tensor] = None,
    ) -> Tensor:
        """
        warm_only=True → L_warm = L_pred + λ_tr L_tr
        否则 → L_joint（式 286）
        """
        w = horizon_weights
        if (not self.use_horizon_weights) or w is None:
            w = None
        loss_pred = prediction_mae_loss(
            pred, target, horizon_weights=w, null_val=self.null_val
        )

        if h_tr is not None and self.lambda_tr > 0.0:
            loss_tr = trend_second_order_smoothness_loss(h_tr)
        else:
            loss_tr = pred.new_zeros(())

        total = loss_pred + self.lambda_tr * loss_tr

        loss_prior_v = pred.new_zeros(())
        loss_str_v = pred.new_zeros(())
        if not warm_only:
            if loss_prior is not None and self.lambda_prior > 0.0:
                loss_prior_v = loss_prior
                total = total + self.lambda_prior * loss_prior_v
            if loss_str is not None and self.lambda_str > 0.0:
                loss_str_v = loss_str
                total = total + self.lambda_str * loss_str_v

        self.last = {
            "pred": float(loss_pred.detach()),
            "tr": float(loss_tr.detach()),
            "prior": float(loss_prior_v.detach()),
            "str": float(loss_str_v.detach()),
            "total": float(total.detach()),
        }
        return total
