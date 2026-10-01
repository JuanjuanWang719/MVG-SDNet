"""
MVGAE / MVGSL 结构预训练（式 76–89）：

  边掩码自监督：E^phy → (E_train, E_val, E_test)；
  每轮 E_vis = E_train \\ E_mask → A_vis^{mv}=F_mv(A_vis^{phy})、X_vis^{str}；
  编码器仅见可见局部结构，在掩码物理边上做关系恢复。
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor
from torch_geometric.utils import negative_sampling

from lib.multiview_graph import (
    MultiViewGraphConfig,
    MultiViewGraphs,
    build_multiview_from_adj_phy,
    build_multiview_graphs,
    build_role_features,
    load_distance_adjacency,
    select_views_for_heads,
)
from lib.joint_losses import kl_beta_warmup
from lib.structure_prior import StructureBasePrior, sparsify_latent_adjacency
from model.PYG_MVGAE_model import build_shared_mvgae


@dataclass
class MVGAEPretrainConfig:
    epochs: int = 300
    lr: float = 1e-3
    latent: int = 32
    num_heads: int = 4  # ≡ num_views
    hidden: Optional[int] = None
    gcn_layers: int = 2
    fusion_dim: Optional[int] = None
    patience: int = 20
    seed: int = 42
    kl_weight: float = 1.0  # β_KL（式 141）
    kl_warmup_epochs: int = 0  # 式(283)：E_KL；0 表示不做 warm-up
    # 视图多样性为可选消融项，式(142)默认不含；置 0
    diversity_weight: float = 0.0
    # 式(86)–(88)：F_s=5 拓扑角色特征
    static_feature_dim: int = 5
    variational: bool = True
    pretrain_data_ratio: float = 1.0
    # 式(77)–(83)：边掩码自监督
    use_edge_mask: bool = True
    mask_ratio: float = 0.2  # ρ_m
    edge_val_ratio: float = 0.1
    edge_test_ratio: float = 0.1
    # 式(121)(125)(127)：重构损失超参
    lambda_neg: float = 1.0
    lambda_mv: float = 1.0
    mv_recon_samples: int = 2048
    # 式(148)–(152)：潜在关系稀疏化与 A^{base}
    latent_top_k: int = 10  # K_p
    latent_tau: float = 0.1  # τ_p
    # 四视图超参
    hop_order: int = 2
    view_hop_k: int = 0
    view_role_k: int = 10
    view_dir_k: int = 10
    sigma_dir: Union[str, float] = "median"
    coords_filename: Optional[str] = None
    phy_distance_weight: bool = True
    sigma_d: Union[str, float] = "median"
    eps_2nd: float = 1e-8


def _as_bool(value, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "y", "on")


def _parse_sigma_dir(raw) -> Union[str, float]:
    if raw is None:
        return "median"
    s = str(raw).strip()
    if s.lower() == "median":
        return "median"
    return float(s)


def mvgae_pretrain_config_from_parser(config) -> MVGAEPretrainConfig:
    defaults = MVGAEPretrainConfig()
    if not config.has_section("MVGAE"):
        return defaults

    section = config["MVGAE"]

    def _opt(name: str, cast, default):
        if config.has_option("MVGAE", name):
            return cast(section[name])
        return default

    hidden = _opt("hidden", int, defaults.hidden) if config.has_option("MVGAE", "hidden") else None
    fusion_dim = (
        _opt("fusion_dim", int, defaults.fusion_dim)
        if config.has_option("MVGAE", "fusion_dim")
        else None
    )
    variational = defaults.variational
    if config.has_option("MVGAE", "variational"):
        variational = _as_bool(section["variational"], default=True)

    pretrain_data_ratio = defaults.pretrain_data_ratio
    if config.has_option("MVGAE", "pretrain_data_ratio"):
        pretrain_data_ratio = float(section["pretrain_data_ratio"])

    coords_filename = None
    if config.has_option("Data", "coords_filename"):
        coords_filename = config["Data"]["coords_filename"]
    elif config.has_option("MVGAE", "coords_filename"):
        coords_filename = section["coords_filename"]

    sigma_dir = defaults.sigma_dir
    if config.has_option("MVGAE", "sigma_dir"):
        sigma_dir = _parse_sigma_dir(section["sigma_dir"])

    sigma_d = defaults.sigma_d
    if config.has_option("MVGAE", "sigma_d"):
        sigma_d = _parse_sigma_dir(section["sigma_d"])

    phy_distance_weight = defaults.phy_distance_weight
    if config.has_option("MVGAE", "phy_distance_weight"):
        phy_distance_weight = _as_bool(section["phy_distance_weight"], default=True)

    use_edge_mask = defaults.use_edge_mask
    if config.has_option("MVGAE", "use_edge_mask"):
        use_edge_mask = _as_bool(section["use_edge_mask"], default=True)

    return MVGAEPretrainConfig(
        epochs=_opt("epochs", int, defaults.epochs),
        lr=_opt("lr", float, defaults.lr),
        latent=_opt("latent", int, defaults.latent),
        num_heads=_opt("num_heads", int, defaults.num_heads),
        hidden=hidden,
        gcn_layers=_opt("gcn_layers", int, defaults.gcn_layers),
        fusion_dim=fusion_dim,
        patience=_opt("patience", int, defaults.patience),
        seed=_opt("seed", int, defaults.seed),
        kl_weight=_opt("kl_weight", float, defaults.kl_weight),
        kl_warmup_epochs=_opt("kl_warmup_epochs", int, defaults.kl_warmup_epochs),
        diversity_weight=_opt("diversity_weight", float, defaults.diversity_weight),
        static_feature_dim=_opt("static_feature_dim", int, defaults.static_feature_dim),
        variational=variational,
        pretrain_data_ratio=pretrain_data_ratio,
        use_edge_mask=use_edge_mask,
        mask_ratio=_opt("mask_ratio", float, defaults.mask_ratio),
        edge_val_ratio=_opt("edge_val_ratio", float, defaults.edge_val_ratio),
        edge_test_ratio=_opt("edge_test_ratio", float, defaults.edge_test_ratio),
        lambda_neg=_opt("lambda_neg", float, defaults.lambda_neg),
        lambda_mv=_opt("lambda_mv", float, defaults.lambda_mv),
        mv_recon_samples=_opt("mv_recon_samples", int, defaults.mv_recon_samples),
        latent_top_k=_opt("latent_top_k", int, defaults.latent_top_k),
        latent_tau=_opt("latent_tau", float, defaults.latent_tau),
        hop_order=_opt("hop_order", int, defaults.hop_order),
        view_hop_k=_opt("view_hop_k", int, defaults.view_hop_k),
        view_role_k=_opt("view_role_k", int, defaults.view_role_k),
        view_dir_k=_opt("view_dir_k", int, defaults.view_dir_k),
        sigma_dir=sigma_dir,
        coords_filename=coords_filename,
        phy_distance_weight=phy_distance_weight,
        sigma_d=sigma_d,
        eps_2nd=_opt("eps_2nd", float, defaults.eps_2nd),
    )


def build_static_node_features(
    adj: np.ndarray,
    num_features: int = 5,
) -> np.ndarray:
    """
    节点结构特征。默认 F_s=5，对齐式(86) 角色向量（已标准化）。
    num_features!=5 时回退到旧的 4 维统计特征（兼容消融）。
    """
    if int(num_features) == 5:
        return build_role_features(adj)

    degree = adj.sum(axis=1)
    neighbor_degree = adj @ degree
    neighbor_degree = neighbor_degree / np.maximum(degree, 1.0)

    adj2 = adj @ adj
    np.fill_diagonal(adj2, 0.0)
    second_order = adj2.sum(axis=1) / np.maximum(degree, 1.0)

    triangles = np.diag(adj @ adj @ adj) / 2.0
    clustering = triangles / np.maximum(degree * (degree - 1.0), 1.0)

    stats = [degree, neighbor_degree, second_order, clustering]
    if num_features > 4:
        stats.extend(
            [
                np.log1p(degree),
                adj.sum(axis=0),
            ]
        )
    features = np.stack(stats[:num_features], axis=1).astype(np.float32)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (features - mean) / std


def subsample_directed_edges(
    edge_index: Tensor,
    ratio: float,
    seed: int,
    edge_weight: Optional[Tensor] = None,
) -> tuple[Tensor, Optional[Tensor]]:
    """按比例随机保留有向边（及对应权重）。"""
    ratio = float(ratio)
    if ratio >= 1.0 - 1e-12:
        return edge_index, edge_weight
    if ratio <= 0.0:
        raise ValueError("pretrain_data_ratio 必须 > 0")
    num_e = int(edge_index.size(1))
    if num_e == 0:
        return edge_index, edge_weight
    keep = max(1, int(round(num_e * ratio)))
    keep = min(keep, num_e)
    g = torch.Generator()
    g.manual_seed(int(seed))
    perm = torch.randperm(num_e, generator=g)[:keep]
    if edge_index.device.type != "cpu":
        perm = perm.to(edge_index.device)
    ei = edge_index[:, perm]
    ew = edge_weight[perm] if edge_weight is not None else None
    return ei, ew


def subsample_bidirected_edges(
    edge_index: Tensor,
    ratio: float,
    seed: int,
    edge_weight: Optional[Tensor] = None,
) -> tuple[Tensor, Optional[Tensor]]:
    """按比例随机保留无向边，再还原为双向 edge_index（权重同步）。"""
    ratio = float(ratio)
    if ratio >= 1.0 - 1e-12:
        return edge_index, edge_weight
    if ratio <= 0.0:
        raise ValueError("pretrain_data_ratio 必须 > 0")

    src, dst = edge_index[0], edge_index[1]
    undirected = src < dst
    u, v = src[undirected], dst[undirected]
    num_u = int(u.numel())
    if num_u == 0:
        return edge_index, edge_weight

    keep = max(1, int(round(num_u * ratio)))
    keep = min(keep, num_u)
    g = torch.Generator()
    g.manual_seed(int(seed))
    perm = torch.randperm(num_u, generator=g)[:keep]
    if u.device.type != "cpu":
        perm = perm.to(u.device)
    u_keep, v_keep = u[perm], v[perm]
    ei = torch.stack(
        [torch.cat([u_keep, v_keep], dim=0), torch.cat([v_keep, u_keep], dim=0)],
        dim=0,
    )
    if edge_weight is None:
        return ei, None
    w_u = edge_weight[undirected][perm]
    ew = torch.cat([w_u, w_u], dim=0)
    return ei, ew


def subsample_view_edges(
    edge_indices: Sequence[Tensor],
    view_names: Sequence[str],
    ratio: float,
    seed: int,
    edge_weights: Optional[Sequence[Optional[Tensor]]] = None,
) -> tuple[list[Tensor], list[Optional[Tensor]]]:
    out_ei: list[Tensor] = []
    out_ew: list[Optional[Tensor]] = []
    for i, (ei, name) in enumerate(zip(edge_indices, view_names)):
        ew = None if edge_weights is None else edge_weights[i]
        if name in ("phy", "2nd", "dir"):
            ei2, ew2 = subsample_directed_edges(ei, ratio, seed + i, ew)
        else:
            ei2, ew2 = subsample_bidirected_edges(ei, ratio, seed + i, ew)
        out_ei.append(ei2)
        out_ew.append(ew2)
    return out_ei, out_ew


def numpy_to_torch(x: np.ndarray, device: torch.device) -> Tensor:
    return torch.from_numpy(x).float().to(device)


def _graph_cfg_from_pretrain(cfg: MVGAEPretrainConfig) -> MultiViewGraphConfig:
    return MultiViewGraphConfig(
        hop_order=cfg.hop_order,
        view_hop_k=cfg.view_hop_k,
        view_role_k=cfg.view_role_k,
        view_dir_k=cfg.view_dir_k,
        sigma_dir=cfg.sigma_dir,
        seed=cfg.seed,
        coords_filename=cfg.coords_filename,
        phy_distance_weight=cfg.phy_distance_weight,
        sigma_d=cfg.sigma_d,
        eps_2nd=cfg.eps_2nd,
    )


def build_views_for_pretrain(
    adj_filename: str,
    num_of_vertices: int,
    id_filename: Optional[str],
    cfg: MVGAEPretrainConfig,
) -> MultiViewGraphs:
    graphs = build_multiview_graphs(
        adj_filename=adj_filename,
        num_of_vertices=num_of_vertices,
        id_filename=id_filename,
        cfg=_graph_cfg_from_pretrain(cfg),
    )
    return select_views_for_heads(graphs, cfg.num_heads)


def phy_edges_from_adj(adj_phy: np.ndarray) -> tuple[Tensor, Tensor]:
    """完整物理有向边 (edge_index [2,E], edge_weight [E])。"""
    src, dst = np.where(adj_phy > 0)
    if src.size == 0:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0,), dtype=torch.float32),
        )
    w = adj_phy[src, dst].astype(np.float32)
    ei = torch.from_numpy(np.stack([src, dst], axis=0)).long()
    return ei, torch.from_numpy(w).float()


def split_phy_edges(
    edge_index: Tensor,
    edge_weight: Tensor,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, Tensor]:
    """式(78)：E^phy = E_train ∪̇ E_val ∪̇ E_test。"""
    num_edges = int(edge_index.size(1))
    if num_edges == 0:
        empty = edge_index
        empty_w = edge_weight
        return {
            "train": empty,
            "train_w": empty_w,
            "val": empty,
            "val_w": empty_w,
            "test": empty,
            "test_w": empty_w,
        }

    val_ratio = max(0.0, float(val_ratio))
    test_ratio = max(0.0, float(test_ratio))
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("edge_val_ratio + edge_test_ratio 必须 < 1")

    g = torch.Generator()
    g.manual_seed(int(seed))
    perm = torch.randperm(num_edges, generator=g)

    n_test = int(math.floor(num_edges * test_ratio))
    n_val = int(math.floor(num_edges * val_ratio))
    if num_edges >= 3:
        n_test = min(n_test, num_edges - 2)
        n_val = min(n_val, num_edges - n_test - 1)
    n_test = max(0, n_test)
    n_val = max(0, n_val)

    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]

    def _take(idx: Tensor) -> tuple[Tensor, Tensor]:
        if idx.numel() == 0:
            return edge_index[:, :0], edge_weight[:0]
        return edge_index[:, idx], edge_weight[idx]

    tr_e, tr_w = _take(train_idx)
    va_e, va_w = _take(val_idx)
    te_e, te_w = _take(test_idx)
    return {
        "train": tr_e,
        "train_w": tr_w,
        "val": va_e,
        "val_w": va_w,
        "test": te_e,
        "test_w": te_w,
    }


def sample_epoch_mask(
    train_edge_index: Tensor,
    mask_ratio: float,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """
    式(79)–(81)：从 E_train 抽 E_mask，返回 (mask_index, vis_index)。
    |E_mask| = floor(ρ_m |E_train|)；尽量保证 E_vis 非空。
    """
    n = int(train_edge_index.size(1))
    if n == 0:
        empty = train_edge_index.new_zeros((0,), dtype=torch.long)
        return empty, empty

    n_mask = int(math.floor(float(mask_ratio) * n))
    if float(mask_ratio) > 0 and n >= 1:
        n_mask = max(1, n_mask)
    else:
        n_mask = 0
    if n > 1:
        n_mask = min(n_mask, n - 1)
    else:
        n_mask = min(n_mask, n)

    perm = torch.randperm(n, generator=generator)
    if train_edge_index.device.type != "cpu":
        perm = perm.to(train_edge_index.device)
    mask_idx = perm[:n_mask]
    vis_idx = perm[n_mask:]
    return mask_idx, vis_idx


def adj_from_edge_subset(
    full_adj: np.ndarray,
    edge_index: Tensor,
) -> np.ndarray:
    """
    式(82)–(83)：仅保留给定有向边；反向边是否存在取决于它是否仍在可见集合中。
    """
    adj = np.zeros_like(full_adj, dtype=np.float32)
    if edge_index.numel() == 0:
        return adj
    src = edge_index[0].detach().cpu().numpy()
    dst = edge_index[1].detach().cpu().numpy()
    adj[src, dst] = full_adj[src, dst]
    return adj


def rebuild_visible_bundle(
    adj_vis: np.ndarray,
    positions: np.ndarray,
    cfg: MVGAEPretrainConfig,
    device: torch.device,
) -> tuple[Tensor, list[Tensor], list[Tensor], np.ndarray]:
    """A_vis^{mv}=F_mv(A_vis^{phy})，并构造标准化 X_vis^{str}。"""
    graphs = build_multiview_from_adj_phy(
        adj_vis, positions, cfg=_graph_cfg_from_pretrain(cfg)
    )
    views = select_views_for_heads(graphs, cfg.num_heads)
    x_np = build_static_node_features(adj_vis, num_features=cfg.static_feature_dim)
    x = numpy_to_torch(x_np, device)
    eis, ews = views.to_device(device)
    return x, eis, ews, x_np


def _split_pos_edges(
    pos_edge_index: Tensor,
    device: torch.device,
    val_ratio: float,
) -> tuple[Tensor, Tensor]:
    num_edges = pos_edge_index.size(1)
    if num_edges == 0:
        empty = pos_edge_index
        return empty, empty
    perm = torch.randperm(num_edges, device=device)
    val_size = max(1, int(num_edges * val_ratio))
    val_size = min(val_size, num_edges)
    val_pos = pos_edge_index[:, perm[:val_size]]
    train_pos = pos_edge_index[:, perm[val_size:]]
    if train_pos.size(1) == 0:
        train_pos = pos_edge_index
        val_pos = pos_edge_index[:, :val_size]
    return train_pos, val_pos


def train_mvgae_masked_pretrain(
    model,
    full_adj: np.ndarray,
    positions: np.ndarray,
    edge_split: dict[str, Tensor],
    device: torch.device,
    cfg: MVGAEPretrainConfig,
    epochs: int = 300,
    lr: float = 1e-3,
    patience: int = 20,
    eval_every: int = 10,
    neg_ratio: int = 1,
    kl_weight: float = 1.0,
    diversity_weight: float = 0.1,
) -> dict:
    """
    式(77)–(89) 掩码自监督训练：
    - 编码器输入仅含 E_vis 上重算的多视图与 X_vis^{str}
    - 在 E_mask 上做物理边关系恢复；E_val/E_test 不参与更新
    """
    num_nodes = int(full_adj.shape[0])
    train_ei = edge_split["train"].to(device)
    val_ei = edge_split["val"].to(device)
    test_ei = edge_split["test"].to(device)

    parts = [p for p in (train_ei, val_ei, test_ei) if p.numel()]
    all_pos = torch.cat(parts, dim=1) if parts else train_ei

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = None
    best_val_auc = -1.0
    counter = 0
    history: list[dict] = []
    gen = torch.Generator()
    gen.manual_seed(int(cfg.seed))

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        mask_idx, vis_idx = sample_epoch_mask(train_ei, cfg.mask_ratio, gen)
        if vis_idx.numel() == 0 and train_ei.size(1) > 0:
            vis_idx = mask_idx[-1:]
            mask_idx = mask_idx[:-1]
        vis_ei = train_ei[:, vis_idx] if vis_idx.numel() else train_ei[:, :0]
        mask_ei = train_ei[:, mask_idx] if mask_idx.numel() else train_ei[:, :0]

        adj_vis = adj_from_edge_subset(full_adj, vis_ei)
        x, eis, ews, _ = rebuild_visible_bundle(adj_vis, positions, cfg, device)

        z = model.encode(x, eis, ews)

        if mask_ei.size(1) == 0:
            pos = vis_ei if vis_ei.size(1) > 0 else train_ei
        else:
            pos = mask_ei
        neg = negative_sampling(
            edge_index=all_pos if all_pos.numel() else pos,
            num_nodes=num_nodes,
            num_neg_samples=max(1, pos.size(1) * neg_ratio),
        )
        # 式(127)：L_rec = L_mask + λ_mv L_mv
        loss_recon, loss_mask, loss_mv = model.structure_recon_loss(
            z,
            pos,
            neg,
            eis,
            ews,
            num_nodes=num_nodes,
            lambda_neg=cfg.lambda_neg,
            lambda_mv=cfg.lambda_mv,
            mv_subsample=cfg.mv_recon_samples,
        )
        # 式(141)(142)：L_pre = L_mask + λ_mv L_mv + β_KL L_KL
        # kl_loss() 已按式(139)对节点取平均，勿再除 N
        loss_kl = model.kl_loss()
        loss_div = model.diversity_loss()
        beta_kl = kl_beta_warmup(epoch, kl_weight, getattr(cfg, "kl_warmup_epochs", 0))
        loss = loss_recon + beta_kl * loss_kl
        if diversity_weight > 0.0:
            loss = loss + diversity_weight * loss_div
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if epoch % eval_every == 0:
            model.eval()
            with torch.no_grad():
                adj_train = adj_from_edge_subset(full_adj, train_ei)
                x_ev, eis_ev, ews_ev, _ = rebuild_visible_bundle(
                    adj_train, positions, cfg, device
                )
                z_eval = model.encode(x_ev, eis_ev, ews_ev)
                if val_ei.size(1) == 0:
                    val_auc, val_ap = 0.0, 0.0
                else:
                    val_neg = negative_sampling(
                        edge_index=all_pos if all_pos.numel() else val_ei,
                        num_nodes=num_nodes,
                        num_neg_samples=val_ei.size(1) * neg_ratio,
                    )
                    val_auc, val_ap = model.test(z_eval, val_ei, val_neg)
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "recon": float(loss_recon.item()),
                    "mask": float(loss_mask.item()),
                    "mv": float(loss_mv.item()),
                    "kl": float(loss_kl.item()),
                    "diversity": float(loss_div.item()),
                    "val_auc": val_auc,
                    "val_ap": val_ap,
                    "n_mask": int(mask_ei.size(1)),
                    "n_vis": int(vis_ei.size(1)),
                }
            )
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    test_auc, test_ap = -1.0, -1.0
    model.eval()
    with torch.no_grad():
        adj_train = adj_from_edge_subset(full_adj, train_ei)
        x_ev, eis_ev, ews_ev, _ = rebuild_visible_bundle(
            adj_train, positions, cfg, device
        )
        z_test = model.encode(x_ev, eis_ev, ews_ev)
        if test_ei.size(1) > 0:
            test_neg = negative_sampling(
                edge_index=all_pos if all_pos.numel() else test_ei,
                num_nodes=num_nodes,
                num_neg_samples=test_ei.size(1) * neg_ratio,
            )
            test_auc, test_ap = model.test(z_test, test_ei, test_neg)

    return {
        "best_val_auc": best_val_auc,
        "test_auc": test_auc,
        "test_ap": test_ap,
        "history": history,
        "num_train_edges": int(train_ei.size(1)),
        "num_val_edges": int(val_ei.size(1)),
        "num_test_edges": int(test_ei.size(1)),
    }


def train_mvgae_pretrain(    model,
    x: Tensor,
    edge_indices: Sequence[Tensor],
    pos_edge_indices: Sequence[Tensor],
    device: torch.device,
    epochs: int = 300,
    lr: float = 1e-3,
    patience: int = 20,
    eval_every: int = 10,
    val_ratio: float = 0.1,
    neg_ratio: int = 1,
    kl_weight: float = 1.0,
    diversity_weight: float = 0.1,
    subsample_size: int = 4096,
    edge_weights: Optional[Sequence[Optional[Tensor]]] = None,
) -> dict:
    """各视图结构重构 + KL + 视图多样性联合优化。"""
    num_nodes = x.size(0)
    num_views = len(pos_edge_indices)
    train_pos_list: list[Tensor] = []
    val_pos_list: list[Tensor] = []
    for pos in pos_edge_indices:
        tr, va = _split_pos_edges(pos, device, val_ratio)
        train_pos_list.append(tr)
        val_pos_list.append(va)

    # 验证优先用物理视图（或第一个非空视图）
    eval_view = 0
    for i, vp in enumerate(val_pos_list):
        if vp.size(1) > 0:
            eval_view = i
            break

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = None
    best_val_auc = -1.0
    counter = 0
    history: list[dict] = []
    ew_list = list(edge_weights) if edge_weights is not None else [None] * num_views

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        z = model.encode(x, list(edge_indices), ew_list)

        pos_batch: list[Tensor] = []
        neg_batch: list[Tensor] = []
        for train_pos in train_pos_list:
            if train_pos.size(1) == 0:
                # 占位：避免空视图崩掉；recon 会返回 0
                empty = train_pos
                pos_batch.append(empty)
                neg_batch.append(empty)
                continue
            if train_pos.size(1) > subsample_size:
                idx = torch.randperm(train_pos.size(1), device=device)[:subsample_size]
                pos = train_pos[:, idx]
            else:
                pos = train_pos
            neg = negative_sampling(
                edge_index=pos if pos.size(1) > 0 else edge_indices[0],
                num_nodes=num_nodes,
                num_neg_samples=max(1, pos.size(1) * neg_ratio),
            )
            pos_batch.append(pos)
            neg_batch.append(neg)

        loss_recon = model.recon_loss(z, pos_batch, neg_batch)
        # 式(139)：kl_loss 已含 1/N 平均
        loss_kl = model.kl_loss()
        loss_div = model.diversity_loss()
        # 无 cfg 时不做 warm-up（兼容旧调用）
        loss = loss_recon + kl_weight * loss_kl
        if diversity_weight > 0.0:
            loss = loss + diversity_weight * loss_div
        loss.backward()
        optimizer.step()

        if epoch % eval_every == 0:
            val_pos = val_pos_list[eval_view]
            if val_pos.size(1) == 0:
                val_auc, val_ap = 0.0, 0.0
            else:
                val_neg = negative_sampling(
                    edge_index=val_pos,
                    num_nodes=num_nodes,
                    num_neg_samples=val_pos.size(1) * neg_ratio,
                )
                z_eval = model.encode(x, list(edge_indices), ew_list)
                val_auc, val_ap = model.test(z_eval, val_pos, val_neg)
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "recon": float(loss_recon.item()),
                    "kl": float(loss_kl.item()),
                    "diversity": float(loss_div.item()),
                    "val_auc": val_auc,
                    "val_ap": val_ap,
                }
            )
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return {"best_val_auc": best_val_auc, "history": history, "num_views": num_views}


def pretrain_mvgae(
    adj_filename: str,
    num_of_vertices: int,
    device: torch.device,
    id_filename: Optional[str] = None,
    cfg: Optional[MVGAEPretrainConfig] = None,
    z_init_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> tuple[np.ndarray, dict]:
    """
    预训练多视图 MVGAE 并导出 Z_init 与模型 checkpoint。

    Returns
    -------
    z_init : [N, fusion_dim]
    stats : dict
    """
    cfg = cfg or MVGAEPretrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    views = build_views_for_pretrain(adj_filename, num_of_vertices, id_filename, cfg)
    full_adj = views.adj_phy.astype(np.float32)
    positions = views.positions
    edge_counts = views.edge_counts()
    hidden_dim = cfg.hidden if cfg.hidden is not None else 2 * cfg.latent
    kl_weight = cfg.kl_weight if cfg.variational else 0.0
    diversity_weight = cfg.diversity_weight if cfg.num_heads >= 2 else 0.0

    phy_ei, phy_ew = phy_edges_from_adj(full_adj)
    if cfg.pretrain_data_ratio < 1.0 - 1e-12:
        phy_ei, phy_ew = subsample_directed_edges(
            phy_ei, cfg.pretrain_data_ratio, cfg.seed, phy_ew
        )
        full_adj = adj_from_edge_subset(full_adj, phy_ei)
        views = select_views_for_heads(
            build_multiview_from_adj_phy(
                full_adj, positions, cfg=_graph_cfg_from_pretrain(cfg)
            ),
            cfg.num_heads,
        )
        edge_counts = views.edge_counts()

    edge_indices_full, edge_weights_full = views.to_device(device)
    static_features = build_static_node_features(
        full_adj, num_features=cfg.static_feature_dim
    )

    print(
        "[MVGSL pretrain] num_views=%d variational=%s latent=%d fusion_dim=%s "
        "mask=%s rho_m=%.3f views=%s"
        % (
            len(views.view_names),
            cfg.variational,
            cfg.latent,
            cfg.fusion_dim,
            cfg.use_edge_mask,
            cfg.mask_ratio,
            ",".join(views.view_names),
        )
    )
    print(
        "[MVGSL pretrain] edge_counts phy=%d 2nd=%d role=%d dir=%d "
        "(phy_asym_pairs=%d dir_asym_pairs=%d)"
        % (
            edge_counts["phy"],
            edge_counts.get("2nd", 0) if "2nd" in views.view_names else 0,
            edge_counts.get("role", 0) if "role" in views.view_names else 0,
            edge_counts.get("dir", 0) if "dir" in views.view_names else 0,
            edge_counts.get("phy_asymmetric_pairs", 0),
            edge_counts.get("dir_asymmetric_pairs", 0),
        )
    )

    model = build_shared_mvgae(
        in_channels=static_features.shape[1],
        hidden_channels=hidden_dim,
        out_channels=cfg.latent,
        num_heads=cfg.num_heads,
        num_gcn_layers=cfg.gcn_layers,
        fusion_out_dim=cfg.fusion_dim,
        variational=cfg.variational,
    ).to(device)

    if cfg.use_edge_mask:
        edge_split = split_phy_edges(
            phy_ei,
            phy_ew if phy_ew is not None else torch.ones(phy_ei.size(1)),
            cfg.edge_val_ratio,
            cfg.edge_test_ratio,
            cfg.seed,
        )
        print(
            "[MVGSL mask] |E_train|=%d |E_val|=%d |E_test|=%d rho_m=%.3f Fs=%d"
            % (
                edge_split["train"].size(1),
                edge_split["val"].size(1),
                edge_split["test"].size(1),
                cfg.mask_ratio,
                static_features.shape[1],
            )
        )
        train_stats = train_mvgae_masked_pretrain(
            model=model,
            full_adj=full_adj,
            positions=positions,
            edge_split=edge_split,
            device=device,
            cfg=cfg,
            epochs=cfg.epochs,
            lr=cfg.lr,
            patience=cfg.patience,
            eval_every=1 if cfg.epochs < 20 else 10,
            kl_weight=kl_weight,
            diversity_weight=diversity_weight,
        )
    else:
        x = numpy_to_torch(static_features, device)
        edge_indices, edge_weights = subsample_view_edges(
            edge_indices_full,
            views.view_names,
            1.0,
            cfg.seed,
            edge_weights_full,
        )
        pos_edge_indices = [ei.clone() for ei in edge_indices]
        train_stats = train_mvgae_pretrain(
            model=model,
            x=x,
            edge_indices=edge_indices,
            pos_edge_indices=pos_edge_indices,
            device=device,
            epochs=cfg.epochs,
            lr=cfg.lr,
            patience=cfg.patience,
            kl_weight=kl_weight,
            diversity_weight=diversity_weight,
            edge_weights=edge_weights,
        )

    x_full = numpy_to_torch(static_features, device)
    z_init = (
        model.compute_z_init(x_full, edge_indices_full, edge_weights_full).cpu().numpy()
    )
    z_s, a_latent, u_node, c_v = model.encode_structure(
        x_full, edge_indices_full, edge_weights_full
    )
    z_s_np = z_s.detach().cpu().numpy().astype(np.float32)
    a_latent_np = a_latent.detach().cpu().numpy().astype(np.float32)
    u_np = u_node.detach().cpu().numpy().astype(np.float32)
    c_np = c_v.detach().cpu().numpy().astype(np.float32)

    # 式(148)–(172)：稀疏 B^{latent}、A^{base}、可信度校准 A^{prior}
    b_latent_np = sparsify_latent_adjacency(
        a_latent_np, top_k=cfg.latent_top_k, tau_p=cfg.latent_tau
    )
    prior = StructureBasePrior(
        a_phy=full_adj,
        b_latent=b_latent_np,
        c_v=c_np,
        top_k=cfg.latent_top_k,
        tau_p=cfg.latent_tau,
        theta_eta_init=0.0,
    )
    a_base_np = prior.base_adjacency().detach().cpu().numpy().astype(np.float32)
    a_prior_np = prior().detach().cpu().numpy().astype(np.float32)
    prior_stats = prior.edge_counts()

    if z_init_path:
        os.makedirs(os.path.dirname(z_init_path) or ".", exist_ok=True)
        np.save(z_init_path, z_init.astype(np.float32))
        base = os.path.splitext(z_init_path)[0]
        np.save(base + "_a_latent.npy", a_latent_np)
        np.save(base + "_uncertainty.npy", u_np)
        np.save(base + "_confidence.npy", c_np)
        np.save(base + "_b_latent.npy", b_latent_np)
        np.save(base + "_a_base.npy", a_base_np)
        np.save(base + "_a_prior.npy", a_prior_np)

    view_adjs = {
        "phy": views.adj_phy,
        "2nd": views.adj_2nd,
        "role": views.adj_role,
        "dir": views.adj_dir,
    }
    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "z_init": z_init,
                "z_s": z_s_np,
                "a_latent": a_latent_np,
                "uncertainty": u_np,
                "confidence": c_np,
                "b_latent": b_latent_np,
                "a_base": a_base_np,
                "a_prior": a_prior_np,
                "a0_prior": a_prior_np,
                "structure_prior": prior_stats,
                "static_features": static_features,
                "config": cfg.__dict__,
                "edge_indices": [ei.cpu() for ei in edge_indices_full],
                "edge_weights": [ew.cpu() for ew in edge_weights_full],
                "view_names": list(views.view_names),
                "view_edge_counts": edge_counts,
                "positions": views.positions,
                "direction_vectors": views.direction_vectors,
                "view_adjs_nnz": {k: int((v > 0).sum()) for k, v in view_adjs.items()},
                "train_stats": {
                    k: train_stats[k]
                    for k in ("best_val_auc", "test_auc", "test_ap")
                    if k in train_stats
                },
            },
            checkpoint_path,
        )

    stats = {
        "best_val_auc": train_stats["best_val_auc"],
        "test_auc": train_stats.get("test_auc", -1.0),
        "z_init_shape": z_init.shape,
        "z_s_shape": z_s_np.shape,
        "a_latent_shape": a_latent_np.shape,
        "uncertainty_shape": u_np.shape,
        "confidence_shape": c_np.shape,
        "b_latent_nnz": prior_stats["b_latent_nnz"],
        "a_base_nnz": prior_stats["a_base_nnz"],
        "a_prior_nnz": prior_stats["a_prior_nnz"],
        "eta_init": prior_stats["eta"],
        "prior_edges": edge_counts["phy"],
        "view_edge_counts": edge_counts,
        "view_names": list(views.view_names),
        "pretrain_data_ratio": float(cfg.pretrain_data_ratio),
        "mask_ratio": float(cfg.mask_ratio),
        "use_edge_mask": bool(cfg.use_edge_mask),
        "z_init_path": z_init_path,
        "checkpoint_path": checkpoint_path,
    }
    return z_init, stats


def load_z_init(
    z_init_path: str,
    device: torch.device,
) -> Tensor:
    z = np.load(z_init_path).astype(np.float32)
    return torch.from_numpy(z).float().to(device)


def _cfg_from_checkpoint_dict(raw: dict, fallback: Optional[MVGAEPretrainConfig] = None) -> MVGAEPretrainConfig:
    """从 checkpoint['config'] 或当前配置重建预训练超参。"""
    base = fallback or MVGAEPretrainConfig()
    if not raw:
        return base

    def _get(name, cast, default):
        if name not in raw or raw[name] is None:
            return default
        return cast(raw[name])

    sigma_dir = raw.get("sigma_dir", base.sigma_dir)
    if not isinstance(sigma_dir, (int, float)):
        sigma_dir = _parse_sigma_dir(sigma_dir)

    return MVGAEPretrainConfig(
        epochs=_get("epochs", int, base.epochs),
        lr=_get("lr", float, base.lr),
        latent=_get("latent", int, base.latent),
        num_heads=_get("num_heads", int, base.num_heads),
        hidden=_get("hidden", int, base.hidden) if raw.get("hidden") is not None else base.hidden,
        gcn_layers=_get("gcn_layers", int, base.gcn_layers),
        fusion_dim=_get("fusion_dim", int, base.fusion_dim) if raw.get("fusion_dim") is not None else base.fusion_dim,
        patience=_get("patience", int, base.patience),
        seed=_get("seed", int, base.seed),
        kl_weight=_get("kl_weight", float, base.kl_weight),
        kl_warmup_epochs=_get("kl_warmup_epochs", int, base.kl_warmup_epochs),
        diversity_weight=_get("diversity_weight", float, base.diversity_weight),
        static_feature_dim=_get("static_feature_dim", int, base.static_feature_dim),
        variational=_as_bool(raw.get("variational"), default=base.variational),
        pretrain_data_ratio=_get("pretrain_data_ratio", float, base.pretrain_data_ratio),
        use_edge_mask=_as_bool(raw.get("use_edge_mask"), default=base.use_edge_mask),
        mask_ratio=_get("mask_ratio", float, base.mask_ratio),
        edge_val_ratio=_get("edge_val_ratio", float, base.edge_val_ratio),
        edge_test_ratio=_get("edge_test_ratio", float, base.edge_test_ratio),
        lambda_neg=_get("lambda_neg", float, base.lambda_neg),
        lambda_mv=_get("lambda_mv", float, base.lambda_mv),
        mv_recon_samples=_get("mv_recon_samples", int, base.mv_recon_samples),
        latent_top_k=_get("latent_top_k", int, base.latent_top_k),
        latent_tau=_get("latent_tau", float, base.latent_tau),
        hop_order=_get("hop_order", int, base.hop_order),
        view_hop_k=_get("view_hop_k", int, base.view_hop_k),
        view_role_k=_get("view_role_k", int, base.view_role_k),
        view_dir_k=_get("view_dir_k", int, base.view_dir_k),
        sigma_dir=sigma_dir,
        coords_filename=raw.get("coords_filename", base.coords_filename),
        phy_distance_weight=_as_bool(
            raw.get("phy_distance_weight"), default=base.phy_distance_weight
        ),
        sigma_d=_parse_sigma_dir(raw.get("sigma_d", base.sigma_d)),
        eps_2nd=_get("eps_2nd", float, base.eps_2nd),
    )


def _resolve_edge_indices(
    ckpt: dict,
    adj_filename: str,
    num_of_vertices: int,
    id_filename: Optional[str],
    cfg: MVGAEPretrainConfig,
    device: torch.device,
) -> tuple[list[Tensor], list[Tensor]]:
    if "edge_indices" in ckpt and ckpt["edge_indices"] is not None:
        eis = [ei.to(device).long() for ei in ckpt["edge_indices"]]
        if len(eis) == cfg.num_heads:
            if "edge_weights" in ckpt and ckpt["edge_weights"] is not None:
                ews = [ew.to(device).float() for ew in ckpt["edge_weights"]]
            else:
                ews = [
                    torch.ones(ei.size(1), device=device, dtype=torch.float32) for ei in eis
                ]
            if len(ews) == len(eis):
                return eis, ews
    views = build_views_for_pretrain(adj_filename, num_of_vertices, id_filename, cfg)
    return views.to_device(device)


def load_mvgae_for_finetune(
    checkpoint_path: str,
    device: torch.device,
    adj_filename: str,
    num_of_vertices: int,
    id_filename: Optional[str] = None,
    cfg: Optional[MVGAEPretrainConfig] = None,
) -> tuple:
    """
    加载预训练 MVGAE 供第二阶段联合微调。

    Returns
    -------
    model, static_x [N, F], edge_indices, edge_weights
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            "联合微调需要 MVGAE checkpoint: %s（请先跑预训练或开启 auto_pretrain）"
            % checkpoint_path
        )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ckpt_cfg = _cfg_from_checkpoint_dict(ckpt.get("config") or {}, fallback=cfg)

    if "static_features" in ckpt and ckpt["static_features"] is not None:
        static_features = np.asarray(ckpt["static_features"], dtype=np.float32)
    else:
        prior_adj = load_distance_adjacency(adj_filename, num_of_vertices, id_filename)
        static_features = build_static_node_features(
            prior_adj, num_features=ckpt_cfg.static_feature_dim
        )

    edge_indices, edge_weights = _resolve_edge_indices(
        ckpt, adj_filename, num_of_vertices, id_filename, ckpt_cfg, device
    )
    x = numpy_to_torch(static_features, device)

    if x.size(0) != num_of_vertices:
        raise ValueError(
            "checkpoint 静态特征节点数 %d 与 num_of_vertices=%d 不一致"
            % (x.size(0), num_of_vertices)
        )

    hidden_dim = ckpt_cfg.hidden if ckpt_cfg.hidden is not None else 2 * ckpt_cfg.latent
    model = build_shared_mvgae(
        in_channels=static_features.shape[1],
        hidden_channels=hidden_dim,
        out_channels=ckpt_cfg.latent,
        num_heads=ckpt_cfg.num_heads,
        num_gcn_layers=ckpt_cfg.gcn_layers,
        fusion_out_dim=ckpt_cfg.fusion_dim,
        variational=ckpt_cfg.variational,
    )
    state = ckpt.get("model_state_dict") or ckpt
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    print(
        "[joint finetune] loaded multi-view MVGAE from %s | views=%d variational=%s fusion_dim=%s"
        % (checkpoint_path, ckpt_cfg.num_heads, ckpt_cfg.variational, model.latent_dim)
    )
    return model, x, edge_indices, edge_weights


def build_mvgae_from_scratch(
    device: torch.device,
    adj_filename: str,
    num_of_vertices: int,
    id_filename: Optional[str] = None,
    cfg: Optional[MVGAEPretrainConfig] = None,
    seed: Optional[int] = None,
) -> tuple:
    """
    端到端联合训练：随机初始化多视图 MVGAE（不加载预训练），与 Hybrid 一起优化。

    Returns
    -------
    model, static_x [N, F], edge_indices, edge_weights
    """
    cfg = cfg or MVGAEPretrainConfig()
    if seed is not None:
        torch.manual_seed(int(seed))

    views = build_views_for_pretrain(adj_filename, num_of_vertices, id_filename, cfg)
    static_features = build_static_node_features(
        views.adj_phy, num_features=cfg.static_feature_dim
    )
    x = numpy_to_torch(static_features, device)
    edge_indices, edge_weights = views.to_device(device)

    hidden_dim = cfg.hidden if cfg.hidden is not None else 2 * cfg.latent
    model = build_shared_mvgae(
        in_channels=static_features.shape[1],
        hidden_channels=hidden_dim,
        out_channels=cfg.latent,
        num_heads=cfg.num_heads,
        num_gcn_layers=cfg.gcn_layers,
        fusion_out_dim=cfg.fusion_dim,
        variational=cfg.variational,
    ).to(device)
    print(
        "[end-to-end] random-init multi-view MVGAE | views=%d variational=%s fusion_dim=%s"
        % (cfg.num_heads, cfg.variational, model.latent_dim)
    )
    return model, x, edge_indices, edge_weights


def transfer_encode_z_init(
    source_checkpoint_path: str,
    target_adj_filename: str,
    target_num_of_vertices: int,
    device: torch.device,
    target_id_filename: Optional[str] = None,
    cfg: Optional[MVGAEPretrainConfig] = None,
    z_init_save_path: Optional[str] = None,
) -> Tensor:
    """
    空间表征迁移：加载源域 MVGAE 编码器权重，用目标域四视图 (A_v,S) 重新编码得到 Z。
    """
    if not os.path.isfile(source_checkpoint_path):
        raise FileNotFoundError(
            "迁移需要源域 MVGAE checkpoint: %s\n"
            "请先在源数据集上完成预训练（例如 PEMS04 完整实验 / train_mvgae_pretrain.py）。"
            % source_checkpoint_path
        )

    ckpt = torch.load(source_checkpoint_path, map_location="cpu")
    ckpt_cfg = _cfg_from_checkpoint_dict(ckpt.get("config") or {}, fallback=cfg)

    views = build_views_for_pretrain(
        target_adj_filename, target_num_of_vertices, target_id_filename, ckpt_cfg
    )
    static_features = build_static_node_features(
        views.adj_phy, num_features=ckpt_cfg.static_feature_dim
    )
    if static_features.shape[0] != target_num_of_vertices:
        raise ValueError(
            "目标图静态特征节点数 %d 与 num_of_vertices=%d 不一致"
            % (static_features.shape[0], target_num_of_vertices)
        )

    x = numpy_to_torch(static_features, device)
    edge_indices, edge_weights = views.to_device(device)

    hidden_dim = ckpt_cfg.hidden if ckpt_cfg.hidden is not None else 2 * ckpt_cfg.latent
    model = build_shared_mvgae(
        in_channels=static_features.shape[1],
        hidden_channels=hidden_dim,
        out_channels=ckpt_cfg.latent,
        num_heads=ckpt_cfg.num_heads,
        num_gcn_layers=ckpt_cfg.gcn_layers,
        fusion_out_dim=ckpt_cfg.fusion_dim,
        variational=ckpt_cfg.variational,
    )
    state = ckpt.get("model_state_dict") or ckpt
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        z_init = model.encode_z_init(x, edge_indices, edge_weights)

    print(
        "[transfer] source=%s → target_nodes=%d | Z shape=%s | views=%d fusion_dim=%s"
        % (
            source_checkpoint_path,
            target_num_of_vertices,
            tuple(z_init.shape),
            ckpt_cfg.num_heads,
            int(z_init.size(1)),
        )
    )

    if z_init_save_path:
        os.makedirs(os.path.dirname(z_init_save_path) or ".", exist_ok=True)
        np.save(z_init_save_path, z_init.detach().cpu().numpy().astype(np.float32))
        print("[transfer] saved Z to", z_init_save_path)

    return z_init
