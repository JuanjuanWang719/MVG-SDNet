"""可配置训练损失（与验证 loss 共用同一 criterion）。"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class CombinedMSEMAELoss(nn.Module):
    """loss = mse_weight * MSE + (1 - mse_weight) * MAE。"""

    def __init__(self, mse_weight: float = 0.5):
        super().__init__()
        if not 0.0 <= mse_weight <= 1.0:
            raise ValueError("mse_weight 应在 [0, 1]")
        self.mse_weight = float(mse_weight)
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        # 最近一次 forward 的分量（detach 标量），便于日志观察量级
        self.last_mse = 0.0
        self.last_mae = 0.0
        self.last_mse_term = 0.0
        self.last_mae_term = 0.0
        self.last_total = 0.0

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_val = self.mse(pred, target)
        mae_val = self.mae(pred, target)
        mse_term = self.mse_weight * mse_val
        mae_term = (1.0 - self.mse_weight) * mae_val
        total = mse_term + mae_term

        self.last_mse = float(mse_val.detach())
        self.last_mae = float(mae_val.detach())
        self.last_mse_term = float(mse_term.detach())
        self.last_mae_term = float(mae_term.detach())
        self.last_total = float(total.detach())
        return total


class MaskedMAELoss(nn.Module):
    """
    验证用 masked MAE。
    compute_val_loss_mstgcn 在 masked_flag=True 时调用
    criterion(outputs, labels, missing_value)。
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        null_val: float = 0.0,
    ) -> torch.Tensor:
        if null_val != null_val:  # NaN
            mask = ~torch.isnan(target)
        else:
            mask = target != float(null_val)
        mask = mask.to(dtype=pred.dtype)
        denom = mask.sum().clamp_min(1.0)
        return ((pred - target).abs() * mask).sum() / denom


def build_criterion(
    loss_name: str,
    device: torch.device,
    mse_weight: float = 0.5,
    huber_beta: float = 1.0,
    masked: bool = False,
) -> nn.Module:
    """
    支持：
      mse / mae(l1) / huber(smooth_l1) / mse_mae / masked_mae
    masked=True 时强制返回 MaskedMAELoss（METR/BAY 验证）。
    """
    if masked:
        return MaskedMAELoss().to(device)
    name = (loss_name or "mse").strip().lower()
    if name == "mse":
        criterion: nn.Module = nn.MSELoss()
    elif name in ("mae", "l1"):
        criterion = nn.L1Loss()
    elif name in ("masked_mae", "mask_mae"):
        criterion = MaskedMAELoss()
    elif name in ("huber", "smooth_l1", "smoothl1"):
        # PyTorch 1.13+：SmoothL1Loss(beta=...)；旧版用 reduction 默认即可
        try:
            criterion = nn.SmoothL1Loss(beta=huber_beta)
        except TypeError:
            criterion = nn.SmoothL1Loss()
    elif name in ("mse_mae", "mse+mae", "mixed"):
        criterion = CombinedMSEMAELoss(mse_weight=mse_weight)
    else:
        raise ValueError(
            "未知 loss_function=%r，可选: mse, mae, huber, mse_mae, masked_mae"
            % (loss_name,)
        )
    return criterion.to(device)

