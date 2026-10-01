#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import argparse
import os

import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from lib.config_io import read_config
from lib.device import resolve_device
from lib.mvgae_pretrain import mvgae_pretrain_config_from_parser, pretrain_mvgae


def main():
    parser = argparse.ArgumentParser(description="Multi-view MVGAE pretrain: export Z_init")
    parser.add_argument("--config", default="configurations/PEMS04_multi_period.conf", type=str)
    parser.add_argument("--z-init-output", type=str, default=None)
    parser.add_argument("--checkpoint-output", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = read_config(args.config)
    data_config = config["Data"]
    cfg = mvgae_pretrain_config_from_parser(config)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.seed is not None:
        cfg.seed = args.seed

    ctx = config["Training"]["ctx"] if config.has_option("Training", "ctx") else "0"
    device = resolve_device(ctx)

    num_of_vertices = int(data_config["num_of_vertices"])
    adj_filename = data_config["adj_filename"]
    id_filename = data_config["id_filename"] if config.has_option("Data", "id_filename") else None

    if args.z_init_output:
        z_init_path = args.z_init_output
    elif config.has_option("Data", "z_init_filename"):
        z_init_path = data_config["z_init_filename"]
    else:
        z_init_path = os.path.join(os.path.dirname(adj_filename), "z_init.npy")

    if args.checkpoint_output:
        checkpoint_path = args.checkpoint_output
    elif config.has_option("Data", "mvgae_checkpoint_filename"):
        checkpoint_path = data_config["mvgae_checkpoint_filename"]
    else:
        checkpoint_path = os.path.join(os.path.dirname(adj_filename), "mvgae_pretrain.pt")

    z_init, stats = pretrain_mvgae(
        adj_filename=adj_filename,
        num_of_vertices=num_of_vertices,
        device=device,
        id_filename=id_filename,
        cfg=cfg,
        z_init_path=z_init_path,
        checkpoint_path=checkpoint_path,
    )

    print(f"Saved Z_init: {z_init_path}, shape={z_init.shape}")
    print(f"Saved checkpoint: {checkpoint_path}")
    print(
        "val_auc=%.4f test_auc=%.4f mask=%s rho_m=%s views=%s"
        % (
            stats["best_val_auc"],
            stats.get("test_auc", -1.0),
            stats.get("use_edge_mask"),
            stats.get("mask_ratio"),
            stats.get("view_names"),
        )
    )


if __name__ == "__main__":
    main()
