"""加载预训练 MVGAE，得到 q(Z|X)=N(μ,σ²) 的节点级 μ、σ（不改训练代码）。"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from lib.config_io import read_config
from lib.mvgae_pretrain import load_mvgae_for_finetune, mvgae_pretrain_config_from_parser
from lib.multiview_graph import load_distance_adjacency
from model.PYG_MVGAE_model import MAX_LOGSTD


def resolve_device(ctx: str) -> torch.device:
    ctx = str(ctx).strip()
    if ctx.startswith("cuda") or ctx.isdigit():
        if not torch.cuda.is_available():
            return torch.device("cpu")
        idx = int(ctx) if ctx.isdigit() else int(ctx.split(":")[-1])
        return torch.device("cuda", idx)
    return torch.device("cpu")


def load_mvgae_posterior(
    config_path: str,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    从 Hybrid/MVGAE 配置加载 checkpoint，encode 得到 μ、σ。

    Returns
    -------
    dict with keys:
      model, mu [N,d], sigma [N,d], logstd [N,d],
      x, edge_indices, edge_weights, a_phy [N,N],
      coords [N,2] or None, meta
    """
    config = read_config(config_path)
    data_cfg = config["Data"]
    train_cfg = config["Training"] if config.has_section("Training") else {}

    num_of_vertices = int(data_cfg["num_of_vertices"])
    adj_filename = data_cfg["adj_filename"]
    id_filename = data_cfg["id_filename"] if config.has_option("Data", "id_filename") else None
    ckpt = checkpoint_path or data_cfg.get("mvgae_checkpoint_filename")
    if not ckpt:
        raise ValueError("配置中缺少 mvgae_checkpoint_filename")

    if device is None:
        ctx = train_cfg.get("ctx", "cpu") if hasattr(train_cfg, "get") else "cpu"
        if config.has_section("Training") and config.has_option("Training", "ctx"):
            ctx = config["Training"]["ctx"]
        device = resolve_device(ctx)

    mvgae_cfg = mvgae_pretrain_config_from_parser(config)
    model, x, edge_indices, edge_weights = load_mvgae_for_finetune(
        checkpoint_path=ckpt,
        device=device,
        adj_filename=adj_filename,
        num_of_vertices=num_of_vertices,
        id_filename=id_filename,
        cfg=mvgae_cfg,
    )
    model.eval()

    with torch.no_grad():
        if not getattr(model, "shared_backbone", False):
            raise RuntimeError("不确定性分析需要 shared_backbone 多视图 MVGAE")
        if not model.variational:
            raise RuntimeError("checkpoint 非 variational，无法得到 σ")
        mu, logstd, _z_s, _hv, _hmv = model.encoder(x, edge_indices, edge_weights)
        logstd = logstd.clamp(max=MAX_LOGSTD)
        sigma = torch.exp(logstd)

    a_phy = load_distance_adjacency(adj_filename, num_of_vertices, id_filename)
    a_phy = (np.asarray(a_phy, dtype=np.float32) > 0).astype(np.float32)

    coords = None
    coords_path = None
    if config.has_option("Data", "coords_filename"):
        coords_path = data_cfg["coords_filename"]
        coords = _load_coords_csv(coords_path, num_of_vertices)

    meta = {
        "config_path": config_path,
        "checkpoint": ckpt,
        "dataset_name": data_cfg.get("dataset_name", ""),
        "num_of_vertices": num_of_vertices,
        "latent_dim": int(mu.size(1)),
        "adj_filename": adj_filename,
        "coords_filename": coords_path,
        "device": str(device),
    }
    return {
        "model": model,
        "mu": mu,
        "sigma": sigma,
        "logstd": logstd,
        "x": x,
        "edge_indices": edge_indices,
        "edge_weights": edge_weights,
        "a_phy": a_phy,
        "coords": coords,
        "meta": meta,
    }


def _load_coords_csv(path: str, n: int) -> np.ndarray:
    import csv
    import os

    if not path or not os.path.isfile(path):
        raise FileNotFoundError("coords_filename 不存在: %s" % path)
    coords = np.zeros((n, 2), dtype=np.float32)
    filled = np.zeros(n, dtype=bool)
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    start = 1 if rows and rows[0] and rows[0][0].lower() in ("id", "node", "index") else 0
    for row in rows[start:]:
        if len(row) < 3:
            continue
        i = int(float(row[0]))
        if 0 <= i < n:
            coords[i, 0] = float(row[1])
            coords[i, 1] = float(row[2])
            filled[i] = True
    if not filled.all():
        raise ValueError("coords 缺少 %d 个节点: %s" % (int((~filled).sum()), path))
    return coords


def sample_latent_z(
    mu: Tensor,
    sigma: Tensor,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """z = μ + σ ⊙ ε, ε~N(0,I)。不依赖 model.training（eval 下 reparametrize 会返回 μ）。"""
    if generator is None:
        eps = torch.randn_like(mu)
    else:
        # Generator 放 CPU，兼容旧版 PyTorch / 多设备
        eps = torch.randn(
            mu.shape, dtype=torch.float32, device="cpu", generator=generator
        ).to(device=mu.device, dtype=mu.dtype)
    return mu + sigma * eps
