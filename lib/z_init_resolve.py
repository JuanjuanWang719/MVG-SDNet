# -*- coding:utf-8 -*-
"""
解析第二阶段节点嵌入 Z 的初始化来源。

- use_node_embed=False：无节点嵌入，仅历史交通序列（消融：seq-only）
- use_mvgae_pretrain=True ：加载 MVGAE 预训练 z_init.npy（可选 auto_pretrain）
- use_mvgae_pretrain=False 且 use_node_embed=True：随机初始化可学习嵌入（消融：w/o MVGAE）
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from torch import Tensor

from lib.mvgae_pretrain import load_z_init, mvgae_pretrain_config_from_parser, pretrain_mvgae


def _as_bool(value, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "y", "on")


def resolve_embed_dim(config, default: int = 48) -> int:
    """随机初始化时的嵌入维数：优先 Hybrid.embed_dim，否则 MVGAE.latent。"""
    if config.has_section("Hybrid") and config.has_option("Hybrid", "embed_dim"):
        return int(config["Hybrid"]["embed_dim"])
    if config.has_section("MVGAE") and config.has_option("MVGAE", "latent"):
        return int(config["MVGAE"]["latent"])
    return default


def make_random_z_init(
    num_nodes: int,
    embed_dim: int,
    device: torch.device,
    seed: Optional[int] = None,
) -> Tensor:
    """与完整模型同形状 [N, D] 的随机节点嵌入（Xavier），供消融公平对比容量。"""
    z = torch.empty(num_nodes, embed_dim, dtype=torch.float32)
    if seed is not None:
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(int(seed))
        try:
            torch.nn.init.xavier_uniform_(z)
        finally:
            torch.random.set_rng_state(rng_state)
    else:
        torch.nn.init.xavier_uniform_(z)
    return z.to(device)


def resolve_use_node_embed(config, cli_override: Optional[bool] = None) -> bool:
    if cli_override is not None:
        return bool(cli_override)
    if config.has_option("Data", "use_node_embed"):
        return _as_bool(config["Data"]["use_node_embed"], default=True)
    if config.has_section("Hybrid") and config.has_option("Hybrid", "use_node_embed"):
        return _as_bool(config["Hybrid"]["use_node_embed"], default=True)
    return True


def resolve_use_mvgae_pretrain(config, cli_override: Optional[bool] = None) -> bool:
    if cli_override is not None:
        return bool(cli_override)
    if config.has_option("Data", "use_mvgae_pretrain"):
        return _as_bool(config["Data"]["use_mvgae_pretrain"], default=True)
    return True


def resolve_z_init(
    config,
    device: torch.device,
    num_of_vertices: int,
    adj_filename: str,
    id_filename: Optional[str] = None,
    seed: Optional[int] = None,
    use_mvgae_pretrain: Optional[bool] = None,
    use_node_embed: Optional[bool] = None,
) -> Tuple[Optional[Tensor], dict]:
    """
    返回 (z_init, info)。use_node_embed=False 时 z_init 为 None。

    info 字段：
      use_node_embed, use_mvgae_pretrain,
      z_init_mode (pretrain|random|none),
      z_init_filename, embed_dim
    """
    data_config = config["Data"]
    use_embed = resolve_use_node_embed(config, use_node_embed)
    use_pretrain = resolve_use_mvgae_pretrain(config, use_mvgae_pretrain)

    z_init_filename = (
        data_config["z_init_filename"]
        if config.has_option("Data", "z_init_filename")
        else os.path.join(os.path.dirname(adj_filename), "z_init.npy")
    )
    auto_pretrain = (
        _as_bool(data_config.get("auto_pretrain"), default=True)
        if config.has_option("Data", "auto_pretrain")
        else True
    )
    embed_dim = resolve_embed_dim(config)

    if not use_embed:
        print(
            "[ablation] use_node_embed=False → 无节点嵌入，"
            "HybridSTPredictor 仅使用历史交通序列（DualScale/StructTemp 可用 z_dim=0）"
        )
        return None, {
            "use_node_embed": False,
            "use_mvgae_pretrain": False,
            "z_init_mode": "none",
            # 仍保留路径，便于加载 A^{prior} 侧车文件（a_latent / confidence）
            "z_init_filename": z_init_filename,
            "embed_dim": 0,
            "auto_pretrain": False,
        }

    # 有节点嵌入时才考虑预训练
    info = {
        "use_node_embed": True,
        "use_mvgae_pretrain": use_pretrain,
        "z_init_mode": "pretrain" if use_pretrain else "random",
        "z_init_filename": z_init_filename if use_pretrain else None,
        "embed_dim": embed_dim,
        "auto_pretrain": auto_pretrain if use_pretrain else False,
    }

    if not use_pretrain:
        print(
            "[ablation] use_mvgae_pretrain=False → 跳过 MVGAE，"
            "随机初始化节点嵌入 Z shape=[%d, %d]" % (num_of_vertices, embed_dim)
        )
        z_init = make_random_z_init(num_of_vertices, embed_dim, device, seed=seed)
        return z_init, info

    if not os.path.isfile(z_init_filename):
        if auto_pretrain:
            print("Z_init not found at %s, running MVGAE pretrain..." % z_init_filename)
            pretrain_cfg = mvgae_pretrain_config_from_parser(config)
            checkpoint_path = (
                data_config["mvgae_checkpoint_filename"]
                if config.has_option("Data", "mvgae_checkpoint_filename")
                else os.path.join(os.path.dirname(adj_filename), "mvgae_pretrain.pt")
            )
            pretrain_mvgae(
                adj_filename=adj_filename,
                num_of_vertices=num_of_vertices,
                device=device,
                id_filename=id_filename,
                cfg=pretrain_cfg,
                z_init_path=z_init_filename,
                checkpoint_path=checkpoint_path,
            )
        else:
            raise FileNotFoundError(
                "Z_init file not found: %s. Run train_mvgae_pretrain.py first, "
                "or set use_mvgae_pretrain=False / use_node_embed=False for ablation."
                % z_init_filename
            )

    z_init = load_z_init(z_init_filename, device)
    if z_init.size(0) != num_of_vertices:
        raise ValueError(
            "z_init 节点数 %d 与 num_of_vertices=%d 不一致：%s"
            % (z_init.size(0), num_of_vertices, z_init_filename)
        )
    info["embed_dim"] = int(z_init.size(1))
    return z_init, info
