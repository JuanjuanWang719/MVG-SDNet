#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations

import argparse
import os
import shutil
from time import time

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from lib.config_io import read_config
from lib.device import resolve_device
from lib.experiment_io import (
    append_runs_summary,
    build_experiment_dir,
    default_run_tag,
    save_results_summary,
    save_run_metadata,
)
from lib.losses import CombinedMSEMAELoss, build_criterion
from lib.joint_losses import JointForecastLoss
from lib.train_stages import apply_stage2_freeze, apply_stage3_unfreeze, build_stage_optimizer
from lib.mvgae_pretrain import (
    build_mvgae_from_scratch,
    load_mvgae_for_finetune,
    load_z_init,
    mvgae_pretrain_config_from_parser,
    pretrain_mvgae,
    transfer_encode_z_init,
)
from lib.utils import (
    compute_val_loss_mstgcn,
    compute_len_input,
    evaluate_on_test_mstgcn,
    get_adjacency_matrix,
    load_graphdata_channel1,
    predict_and_save_results_mstgcn,
    set_seed,
)
from lib.z_init_resolve import resolve_z_init
from lib.z_similarity import compute_z_init_similarity_to_baseline
from lib.structure_prior import StructureBasePrior, sparsify_latent_adjacency
from model.HybridSTPredictor import make_hybrid_model, normalize_fusion_mode


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    default="configurations/PEMS04_multi_period.conf",
    type=str,
    help="configuration file path",
)
parser.add_argument("--epochs", type=int, default=None, help="覆盖配置文件中的 epochs")
parser.add_argument("--start-epoch", type=int, default=None, help="覆盖配置文件中的 start_epoch")
parser.add_argument("--patience", type=int, default=None, help="覆盖配置文件中的 patience（0=禁用早停）")
parser.add_argument("--learning-rate", type=float, default=None, help="覆盖配置文件中的 learning_rate")
parser.add_argument("--z-lr-scale", type=float, default=None, help="覆盖 [Hybrid] z_lr_scale")
parser.add_argument("--model-name", type=str, default=None, help="覆盖 [Training] model_name（用于区分 sweep 实验目录）")
parser.add_argument("--seed", type=int, default=None, help="覆盖配置文件中的 seed（固定随机性）")
parser.add_argument(
    "--loss-function",
    type=str,
    default=None,
    help="覆盖损失: mse / mae / huber / mse_mae",
)
parser.add_argument(
    "--lr-scheduler",
    type=str,
    default=None,
    help="覆盖学习率调度: none / cosine",
)
parser.add_argument("--run-tag", type=str, default=None, help="实验子目录标签（默认自动生成时间戳，避免覆盖旧结果）")
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="若目标实验目录已存在则先删除（默认保留历史实验，每次使用新目录）",
)
parser.add_argument(
    "--no-mvgae-pretrain",
    action="store_true",
    help="消融：跳过 MVGAE，随机初始化节点嵌入（覆盖配置 use_mvgae_pretrain）",
)
parser.add_argument(
    "--use-mvgae-pretrain",
    action="store_true",
    help="强制使用 MVGAE 预训练 Z_init（覆盖配置）",
)
parser.add_argument(
    "--no-node-embed",
    action="store_true",
    help="消融：完全关闭节点嵌入，仅用历史交通序列（覆盖配置 use_node_embed）",
)
args = parser.parse_args()
if args.no_mvgae_pretrain and args.use_mvgae_pretrain:
    raise SystemExit("不能同时指定 --no-mvgae-pretrain 与 --use-mvgae-pretrain")
if args.no_node_embed and args.use_mvgae_pretrain:
    raise SystemExit("不能同时指定 --no-node-embed 与 --use-mvgae-pretrain")
config = read_config(args.config)
data_config = config["Data"]
training_config = config["Training"]
hybrid_config = config["Hybrid"] if config.has_section("Hybrid") else {}

graph_signal_matrix_filename = data_config["graph_signal_matrix_filename"]
adj_filename = data_config["adj_filename"]
num_of_vertices = int(data_config["num_of_vertices"])
num_for_predict = int(data_config["num_for_predict"])
len_input = int(data_config["len_input"])
dataset_name = data_config["dataset_name"]

if config.has_option("Data", "id_filename"):
    id_filename = data_config["id_filename"]
else:
    id_filename = None

cli_use_mvgae = None
if args.no_mvgae_pretrain:
    cli_use_mvgae = False
elif args.use_mvgae_pretrain:
    cli_use_mvgae = True
cli_use_node_embed = False if args.no_node_embed else None

model_name = training_config.get("model_name", "hybrid_st")
if args.model_name is not None:
    model_name = args.model_name
ctx = training_config["ctx"]
DEVICE = resolve_device(ctx)
USE_CUDA = DEVICE.type == "cuda"

if config.has_option("Training", "seed"):
    seed = int(training_config["seed"])
elif config.has_option("MVGAE", "seed"):
    seed = int(config["MVGAE"]["seed"])
else:
    seed = 42
if args.seed is not None:
    seed = args.seed
cudnn_deterministic = False
if config.has_option("Training", "cudnn_deterministic"):
    cudnn_deterministic = training_config.get("cudnn_deterministic", "False").lower() in (
        "true",
        "1",
        "yes",
    )
set_seed(seed, deterministic=cudnn_deterministic)
print("random seed:", seed)
print("cudnn_deterministic:", cudnn_deterministic, "(True 更慢但更易复现)")

learning_rate = float(training_config["learning_rate"])
if args.learning_rate is not None:
    learning_rate = args.learning_rate
epochs = int(training_config["epochs"])
start_epoch = int(training_config["start_epoch"])
if args.epochs is not None:
    epochs = args.epochs
if args.start_epoch is not None:
    start_epoch = args.start_epoch
batch_size = int(training_config["batch_size"])
num_of_weeks = int(training_config["num_of_weeks"])
num_of_days = int(training_config["num_of_days"])
num_of_hours = int(training_config["num_of_hours"])
in_channels = int(training_config["in_channels"])
missing_value = float(training_config["missing_value"]) if config.has_option("Training", "missing_value") else 0.0
metric_method = (
    training_config.get("metric_method", "unmask").strip().lower()
    if config.has_option("Training", "metric_method")
    else "unmask"
)
masked_flag = 1 if metric_method == "mask" else 0
train_null_val = float(missing_value) if masked_flag else None

loss_function = training_config.get("loss_function", "mse")
if args.loss_function is not None:
    loss_function = args.loss_function
mse_weight = float(training_config.get("mse_weight", 0.5)) if config.has_option("Training", "mse_weight") else 0.5
huber_beta = float(training_config.get("huber_beta", 1.0)) if config.has_option("Training", "huber_beta") else 1.0

lr_scheduler_name = training_config.get("lr_scheduler", "none")
if args.lr_scheduler is not None:
    lr_scheduler_name = args.lr_scheduler
lr_scheduler_name = lr_scheduler_name.strip().lower()
cosine_eta_min = (
    float(training_config.get("cosine_eta_min", 1e-6))
    if config.has_option("Training", "cosine_eta_min")
    else 1e-6
)
if config.has_option("Training", "cosine_t_max"):
    cosine_t_max = int(training_config["cosine_t_max"])
else:
    cosine_t_max = None  # 默认用 epochs - start_epoch

expected_len = compute_len_input(num_of_weeks, num_of_days, num_of_hours, num_for_predict)
if len_input != expected_len:
    raise ValueError(
        f"len_input={len_input} 与多周期配置不一致，应为 {expected_len}；"
        f"请检查配置或重新运行 prepareData.py"
    )

local_hidden = int(hybrid_config.get("local_hidden", 64))
global_d_model = int(hybrid_config.get("global_d_model", 64))
global_nhead = int(hybrid_config.get("global_nhead", 4))
global_layers = int(hybrid_config.get("global_layers", 2))
gwnet_blocks = int(hybrid_config.get("gwnet_blocks", hybrid_config.get("local_layers", 4)))
gwnet_layers = int(hybrid_config.get("gwnet_layers", 2))
gwnet_nhid = int(hybrid_config.get("gwnet_nhid", 32))
gcn_bool = hybrid_config.get("gcn_bool", "True").lower() in ("true", "1", "yes")
addaptadj = hybrid_config.get("addaptadj", "True").lower() in ("true", "1", "yes")
fusion_dim = int(hybrid_config.get("fusion_dim", 64))
dropout = float(hybrid_config.get("dropout", 0.1))
gwnet_dropout = float(hybrid_config.get("gwnet_dropout", 0.3))
z_lr_scale = float(hybrid_config.get("z_lr_scale", 0.1))
if args.z_lr_scale is not None:
    z_lr_scale = args.z_lr_scale
prior_loss_weight = float(hybrid_config.get("prior_loss_weight", 0.1))
lambda_tr = float(hybrid_config.get("lambda_tr", 0.01))
lambda_str = float(hybrid_config.get("lambda_str", 0.0))
warm_epochs = int(hybrid_config.get("warm_epochs", 20))
use_three_stage = hybrid_config.get("use_three_stage", "True").lower() in (
    "true",
    "1",
    "yes",
)
str_lr_scale = float(hybrid_config.get("str_lr_scale", z_lr_scale))
grad_clip = float(hybrid_config.get("grad_clip", 5.0))
eval_test_every = int(hybrid_config.get("eval_test_every", 1))
log_every = int(hybrid_config.get("log_every", 100))
patience = int(hybrid_config.get("patience", 15))
if args.patience is not None:
    patience = args.patience
period_split = hybrid_config.get("period_split", "True").lower() in ("true", "1", "yes")
fusion_mode = normalize_fusion_mode(hybrid_config.get("fusion_mode", "gated"))
use_local_branch = hybrid_config.get("use_local_branch", "True").lower() in ("true", "1", "yes")
use_global_branch = hybrid_config.get("use_global_branch", "True").lower() in ("true", "1", "yes")
use_dual_scale_spatial = hybrid_config.get("use_dual_scale_spatial", "False").lower() in (
    "true",
    "1",
    "yes",
)
dyn_top_k = int(hybrid_config.get("dyn_top_k", 10))
dyn_attn_dim = int(hybrid_config.get("dyn_attn_dim", 32))
dyn_state_dim = int(hybrid_config.get("dyn_state_dim", 64))
dyn_sp_layers = int(hybrid_config.get("dyn_sp_layers", 2))
d_tod = int(hybrid_config.get("d_tod", 16))
d_dow = int(hybrid_config.get("d_dow", 8))
points_per_hour = int(data_config.get("points_per_hour", 12))
steps_per_day = int(hybrid_config.get("steps_per_day", points_per_hour * 24))
use_structure_aware_temporal = hybrid_config.get(
    "use_structure_aware_temporal", "False"
).lower() in ("true", "1", "yes")
tem_hidden = int(hybrid_config.get("tem_hidden", 64))
tem_gru_layers = int(hybrid_config.get("tem_gru_layers", 1))
tem_n_heads = int(hybrid_config.get("tem_n_heads", 4))
tem_d_model = int(hybrid_config.get("tem_d_model", 64))
joint_finetune_mvgae = hybrid_config.get("joint_finetune_mvgae", "False").lower() in (
    "true",
    "1",
    "yes",
)
end_to_end_mvgae = hybrid_config.get("end_to_end_mvgae", "False").lower() in (
    "true",
    "1",
    "yes",
)
transfer_encoder = False
if config.has_option("Data", "transfer_encoder"):
    transfer_encoder = data_config.get("transfer_encoder", "False").lower() in (
        "true",
        "1",
        "yes",
    )
elif config.has_option("Hybrid", "transfer_encoder"):
    transfer_encoder = hybrid_config.get("transfer_encoder", "False").lower() in (
        "true",
        "1",
        "yes",
    )
_mode_flags = [joint_finetune_mvgae, end_to_end_mvgae, transfer_encoder]
if sum(1 for x in _mode_flags if x) > 1:
    raise SystemExit(
        "joint_finetune_mvgae / end_to_end_mvgae / transfer_encoder 最多只能开启一个"
    )
_default_mvgae_lr_scale = 1.0 if end_to_end_mvgae else z_lr_scale
mvgae_lr_scale = float(hybrid_config.get("mvgae_lr_scale", _default_mvgae_lr_scale))
if args.no_node_embed and (joint_finetune_mvgae or end_to_end_mvgae or transfer_encoder):
    raise SystemExit("MVGAE 联合/端到端/迁移需要节点嵌入，不能与 --no-node-embed 同时使用")
if args.no_mvgae_pretrain and joint_finetune_mvgae:
    raise SystemExit("联合微调 MVGAE 需要预训练权重，不能与 --no-mvgae-pretrain 同时使用")

source_mvgae_checkpoint = None
if transfer_encoder:
    if config.has_option("Data", "source_mvgae_checkpoint"):
        source_mvgae_checkpoint = data_config["source_mvgae_checkpoint"]
    else:
        raise SystemExit(
            "transfer_encoder=True 时必须在 [Data] 指定 source_mvgae_checkpoint（源域 mvgae_pretrain.pt）"
        )

run_tag = args.run_tag if args.run_tag else default_run_tag()

if joint_finetune_mvgae or end_to_end_mvgae:
    folder_dir = "%s_h%dd%dw%d_channel%d_lr%g_mvgae%g" % (
        model_name,
        num_of_hours,
        num_of_days,
        num_of_weeks,
        in_channels,
        learning_rate,
        mvgae_lr_scale,
    )
else:
    folder_dir = "%s_h%dd%dw%d_channel%d_lr%g_z%g" % (
        model_name,
        num_of_hours,
        num_of_days,
        num_of_weeks,
        in_channels,
        learning_rate,
        z_lr_scale,
    )
params_path = str(build_experiment_dir(dataset_name, folder_dir, run_tag))

train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _mean, _std = (
    load_graphdata_channel1(
        graph_signal_matrix_filename,
        num_of_hours,
        num_of_days,
        num_of_weeks,
        DEVICE,
        batch_size,
        seed=seed,
    )
)

mvgae_module = None
mvgae_x = None
mvgae_edge_indices = None
mvgae_edge_weights = None
preserve_mvgae_weights = True

if transfer_encoder:
    print("[transfer] encoder frozen; encode target graph with source weights")
    z_cache = (
        data_config["z_init_filename"]
        if config.has_option("Data", "z_init_filename")
        else None
    )
    force_reencode = False
    if config.has_option("Data", "force_reencode_transfer"):
        force_reencode = data_config.get("force_reencode_transfer", "False").lower() in (
            "true",
            "1",
            "yes",
        )
    if z_cache and os.path.isfile(z_cache) and not force_reencode:
        z_init = load_z_init(z_cache, DEVICE)
        print("[transfer] load cached Z from", z_cache, "shape=", tuple(z_init.shape))
        if z_init.size(0) != num_of_vertices:
            raise ValueError(
                "缓存 Z 节点数 %d 与目标 num_of_vertices=%d 不一致，请删缓存或设 force_reencode_transfer=True"
                % (z_init.size(0), num_of_vertices)
            )
    else:
        z_init = transfer_encode_z_init(
            source_checkpoint_path=source_mvgae_checkpoint,
            target_adj_filename=adj_filename,
            target_num_of_vertices=num_of_vertices,
            device=DEVICE,
            target_id_filename=id_filename,
            cfg=mvgae_pretrain_config_from_parser(config),
            z_init_save_path=z_cache,
        )
    adj_mx, _ = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)
    if adj_mx.shape[0] != num_of_vertices:
        raise ValueError(
            f"邻接矩阵形状 {adj_mx.shape} 与 num_of_vertices={num_of_vertices} 不一致：{adj_filename}"
        )
    z_init_info = {
        "use_node_embed": True,
        "use_mvgae_pretrain": False,
        "z_init_mode": "transfer",
        "z_init_filename": z_cache,
        "embed_dim": int(z_init.size(1)),
        "joint_finetune_mvgae": False,
        "end_to_end_mvgae": False,
        "transfer_encoder": True,
        "source_mvgae_checkpoint": os.path.abspath(source_mvgae_checkpoint),
    }
elif end_to_end_mvgae:
    print("[end-to-end] skip MVGAE pretrain / z_init.npy；随机初始化编码器")
    pretrain_cfg = mvgae_pretrain_config_from_parser(config)
    mvgae_module, mvgae_x, mvgae_edge_indices, mvgae_edge_weights = build_mvgae_from_scratch(
        device=DEVICE,
        adj_filename=adj_filename,
        num_of_vertices=num_of_vertices,
        id_filename=id_filename,
        cfg=pretrain_cfg,
        seed=seed,
    )
    adj_mx, _ = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)
    if adj_mx.shape[0] != num_of_vertices:
        raise ValueError(
            f"邻接矩阵形状 {adj_mx.shape} 与 num_of_vertices={num_of_vertices} 不一致：{adj_filename}"
        )
    z_init = None
    z_init_info = {
        "use_node_embed": True,
        "use_mvgae_pretrain": False,
        "z_init_mode": "end_to_end",
        "z_init_filename": None,
        "embed_dim": int(mvgae_module.latent_dim),
        "joint_finetune_mvgae": False,
        "end_to_end_mvgae": True,
        "transfer_encoder": False,
    }
    preserve_mvgae_weights = False
else:
    z_init, z_init_info = resolve_z_init(
        config=config,
        device=DEVICE,
        num_of_vertices=num_of_vertices,
        adj_filename=adj_filename,
        id_filename=id_filename,
        seed=seed,
        use_mvgae_pretrain=True if joint_finetune_mvgae else cli_use_mvgae,
        use_node_embed=True if joint_finetune_mvgae else cli_use_node_embed,
    )
    adj_mx, _ = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)
    if adj_mx.shape[0] != num_of_vertices:
        raise ValueError(
            f"邻接矩阵形状 {adj_mx.shape} 与 num_of_vertices={num_of_vertices} 不一致：{adj_filename}"
        )
    z_init_info["end_to_end_mvgae"] = False
    z_init_info["transfer_encoder"] = False
    if joint_finetune_mvgae:
        if not z_init_info.get("use_node_embed", True):
            raise ValueError("joint_finetune_mvgae=True 时必须 use_node_embed=True")
        checkpoint_path = (
            data_config["mvgae_checkpoint_filename"]
            if config.has_option("Data", "mvgae_checkpoint_filename")
            else os.path.join(os.path.dirname(adj_filename), "mvgae_pretrain.pt")
        )
        if not os.path.isfile(checkpoint_path):
            print("MVGAE checkpoint missing, re-running pretrain for joint finetune:", checkpoint_path)
            pretrain_cfg = mvgae_pretrain_config_from_parser(config)
            z_path = z_init_info.get("z_init_filename") or os.path.join(
                os.path.dirname(adj_filename), "z_init.npy"
            )
            pretrain_mvgae(
                adj_filename=adj_filename,
                num_of_vertices=num_of_vertices,
                device=DEVICE,
                id_filename=id_filename,
                cfg=pretrain_cfg,
                z_init_path=z_path,
                checkpoint_path=checkpoint_path,
            )
        mvgae_module, mvgae_x, mvgae_edge_indices, mvgae_edge_weights = load_mvgae_for_finetune(
            checkpoint_path=checkpoint_path,
            device=DEVICE,
            adj_filename=adj_filename,
            num_of_vertices=num_of_vertices,
            id_filename=id_filename,
            cfg=mvgae_pretrain_config_from_parser(config),
        )
        z_init = None
        z_init_info["joint_finetune_mvgae"] = True
        z_init_info["z_init_mode"] = "joint_mvgae"
        z_init_info["embed_dim"] = int(mvgae_module.latent_dim)
        preserve_mvgae_weights = True
    else:
        z_init_info["joint_finetune_mvgae"] = False

z_sim_info = None
z_sim_baseline = None
if config.has_option("Data", "z_init_similarity_baseline"):
    z_sim_baseline = data_config.get("z_init_similarity_baseline", "").strip() or None
z_sim_info = compute_z_init_similarity_to_baseline(z_init, z_sim_baseline)
if z_sim_info:
    z_init_info.update(z_sim_info)

# 式(148)–(152)：可选 A^{base} 结构先验（可学习 η）
structure_prior = None
use_abase_prior = hybrid_config.get("use_abase_prior", "True").lower() in (
    "true",
    "1",
    "yes",
)
latent_top_k = int(
    config["MVGAE"].get("latent_top_k", 10)
    if config.has_section("MVGAE")
    else hybrid_config.get("latent_top_k", 10)
)
latent_tau = float(
    config["MVGAE"].get("latent_tau", 0.1)
    if config.has_section("MVGAE")
    else hybrid_config.get("latent_tau", 0.1)
)
a_latent_path = None
b_latent_path = None
z_path = z_init_info.get("z_init_filename")
if z_path:
    a_latent_path = os.path.splitext(z_path)[0] + "_a_latent.npy"
    b_latent_path = os.path.splitext(z_path)[0] + "_b_latent.npy"

if use_abase_prior:
    if a_latent_path and os.path.isfile(a_latent_path):
        a_lat = np.load(a_latent_path).astype(np.float32)
        if b_latent_path and os.path.isfile(b_latent_path):
            b_lat = np.load(b_latent_path).astype(np.float32)
        else:
            b_lat = sparsify_latent_adjacency(a_lat, top_k=latent_top_k, tau_p=latent_tau)
        c_path = os.path.splitext(z_path)[0] + "_confidence.npy" if z_path else None
        c_vec = None
        if c_path and os.path.isfile(c_path):
            c_vec = np.load(c_path).astype(np.float32)
        structure_prior = StructureBasePrior(
            a_phy=adj_mx.astype(np.float32),
            b_latent=b_lat,
            c_v=c_vec,
            top_k=latent_top_k,
            tau_p=latent_tau,
            theta_eta_init=0.0,
        )
        print(
            "[A_prior] enabled | K_p=%d tau=%.3f B_nnz=%d eta0=%.3f c_mean=%.3f"
            % (
                latent_top_k,
                latent_tau,
                int((b_lat > 0).sum()),
                float(structure_prior.eta.item()),
                float(structure_prior.c_v.mean().item()),
            )
        )
    else:
        print(
            "[A_prior] 未找到 a_latent（path=%s）；"
            "若启用 DualScale 将回退为物理图锚点"
            % (a_latent_path,)
        )

def _build_phy_only_prior():
    n = int(adj_mx.shape[0])
    prior = StructureBasePrior(
        a_phy=adj_mx.astype(np.float32),
        b_latent=np.zeros((n, n), dtype=np.float32),
        c_v=np.ones(n, dtype=np.float32),
        top_k=latent_top_k,
        tau_p=latent_tau,
        theta_eta_init=10.0,  # sigmoid(10)≈1 → A^{base}≈A^{phy}
    )
    prior.theta_eta.requires_grad_(False)
    return prior

# DualScale 必须有 prior：无 a_latent / 显式关闭 abase 时用 phy-only
if structure_prior is None and use_dual_scale_spatial:
    structure_prior = _build_phy_only_prior()
    print("[A_prior] phy-only anchor | DualScale keeps A_dyn fusion")

if use_dual_scale_spatial and structure_prior is None:
    print("[DualScale] disabled：需要 A^{prior}（请先完成结构先验导出）")
    use_dual_scale_spatial = False

net = make_hybrid_model(
    device=DEVICE,
    adj_mx=adj_mx,
    z_init=z_init,
    use_node_embed=z_init_info["use_node_embed"],
    num_nodes=num_of_vertices,
    in_channels=in_channels,
    seq_len=len_input,
    num_for_predict=num_for_predict,
    local_hidden=local_hidden,
    global_d_model=global_d_model,
    global_nhead=global_nhead,
    global_layers=global_layers,
    gwnet_blocks=gwnet_blocks,
    gwnet_layers=gwnet_layers,
    gwnet_nhid=gwnet_nhid,
    gcn_bool=gcn_bool,
    addaptadj=addaptadj,
    fusion_dim=fusion_dim,
    dropout=dropout,
    gwnet_dropout=gwnet_dropout,
    num_of_weeks=num_of_weeks,
    num_of_days=num_of_days,
    num_of_hours=num_of_hours,
    period_split=period_split,
    fusion_mode=fusion_mode,
    use_local_branch=use_local_branch,
    use_global_branch=use_global_branch,
    mvgae=mvgae_module,
    mvgae_x=mvgae_x,
    mvgae_edge_indices=mvgae_edge_indices,
    mvgae_edge_weights=mvgae_edge_weights,
    preserve_mvgae_weights=preserve_mvgae_weights,
    structure_prior=structure_prior,
    use_dual_scale_spatial=use_dual_scale_spatial,
    dyn_top_k=dyn_top_k,
    dyn_attn_dim=dyn_attn_dim,
    dyn_state_dim=dyn_state_dim,
    dyn_sp_layers=dyn_sp_layers,
    d_tod=d_tod,
    d_dow=d_dow,
    steps_per_day=steps_per_day,
    use_structure_aware_temporal=use_structure_aware_temporal,
    tem_hidden=tem_hidden,
    tem_gru_layers=tem_gru_layers,
    tem_n_heads=tem_n_heads,
    tem_d_model=tem_d_model,
)


def train_main():
    if os.path.exists(params_path):
        if args.overwrite and start_epoch == 0:
            try:
                shutil.rmtree(params_path)
            except PermissionError:
                print("warning: cannot remove existing params directory (in use), reusing it")
        elif start_epoch == 0:
            print("experiment directory already exists, reusing:", params_path)
        else:
            print("train from params directory %s" % (params_path))
    else:
        os.makedirs(params_path, exist_ok=True)
        print("create params directory %s" % (params_path))

    save_run_metadata(
        params_path,
        args.config,
        {
            "run_tag": run_tag,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "seed": seed,
            "learning_rate": learning_rate,
            "z_lr_scale": z_lr_scale,
            "epochs": epochs,
            "patience": patience,
            "len_input": len_input,
            "num_of_weeks": num_of_weeks,
            "num_of_days": num_of_days,
            "num_of_hours": num_of_hours,
            "period_split": period_split,
            "fusion_mode": fusion_mode,
            "use_local_branch": use_local_branch,
            "use_global_branch": use_global_branch,
            "joint_finetune_mvgae": joint_finetune_mvgae,
            "end_to_end_mvgae": end_to_end_mvgae,
            "transfer_encoder": transfer_encoder,
            "source_mvgae_checkpoint": source_mvgae_checkpoint,
            "mvgae_lr_scale": mvgae_lr_scale if (joint_finetune_mvgae or end_to_end_mvgae) else None,
            "local_hidden": local_hidden,
            "global_d_model": global_d_model,
            "global_layers": global_layers,
            "gwnet_blocks": gwnet_blocks,
            "gwnet_layers": gwnet_layers,
            "gwnet_nhid": gwnet_nhid,
            "fusion_dim": fusion_dim,
            "dropout": dropout,
            "gwnet_dropout": gwnet_dropout,
            "use_mvgae_pretrain": z_init_info["use_mvgae_pretrain"],
            "use_node_embed": z_init_info["use_node_embed"],
            "z_init_mode": z_init_info["z_init_mode"],
            "embed_dim": z_init_info["embed_dim"],
            "z_init_filename": z_init_info.get("z_init_filename"),
            "z_init_sim_to_r100": z_init_info.get("z_init_sim_to_r100"),
            "z_init_similarity_baseline": z_init_info.get("z_init_similarity_baseline"),
            "loss_function": loss_function,
            "mse_weight": mse_weight,
            "huber_beta": huber_beta,
            "lr_scheduler": lr_scheduler_name,
            "cosine_eta_min": cosine_eta_min,
            "cosine_t_max": cosine_t_max if cosine_t_max is not None else (epochs - start_epoch),
            "params_path": os.path.abspath(params_path),
        },
    )

    print("param list:")
    print("params_path\t", params_path)
    print("run_tag\t", run_tag)
    print("CUDA\t", DEVICE)
    print("seed\t", seed)
    print("use_mvgae_pretrain\t", z_init_info["use_mvgae_pretrain"])
    print("use_node_embed\t", z_init_info["use_node_embed"])
    print("z_init_mode\t", z_init_info["z_init_mode"])
    print("embed_dim\t", z_init_info["embed_dim"])
    print("z_init_filename\t", z_init_info.get("z_init_filename"))
    if z_init_info.get("z_init_sim_to_r100") is not None:
        print("z_init_sim_to_r100\t", "%.6f" % z_init_info["z_init_sim_to_r100"])
        print("z_init_similarity_baseline\t", z_init_info.get("z_init_similarity_baseline"))
    print("in_channels\t", in_channels)
    print("len_input\t", len_input)
    print("num_of_weeks\t", num_of_weeks)
    print("num_of_days\t", num_of_days)
    print("num_of_hours\t", num_of_hours)
    print("period_split\t", period_split)
    print("fusion_mode\t", fusion_mode)
    print("use_local_branch\t", use_local_branch)
    print("use_global_branch\t", use_global_branch)
    print("use_dual_scale_spatial\t", use_dual_scale_spatial)
    if use_dual_scale_spatial:
        print("dyn_top_k\t", dyn_top_k)
        print("dyn_state_dim\t", dyn_state_dim)
        print("dyn_sp_layers\t", dyn_sp_layers)
    print("use_structure_aware_temporal\t", use_structure_aware_temporal)
    if use_structure_aware_temporal:
        print("tem_hidden\t", tem_hidden)
        print("tem_gru_layers\t", tem_gru_layers)
        print("tem_n_heads\t", tem_n_heads)
    print("joint_finetune_mvgae\t", joint_finetune_mvgae)
    print("end_to_end_mvgae\t", end_to_end_mvgae)
    print("transfer_encoder\t", transfer_encoder)
    if transfer_encoder:
        print("source_mvgae_checkpoint\t", source_mvgae_checkpoint)
    if joint_finetune_mvgae or end_to_end_mvgae:
        print("mvgae_lr_scale\t", mvgae_lr_scale)
    print("local_hidden\t", local_hidden)
    print("gwnet_blocks\t", gwnet_blocks)
    print("gwnet_layers\t", gwnet_layers)
    print("gwnet_nhid\t", gwnet_nhid)
    print("global_d_model\t", global_d_model)
    print("learning_rate\t", learning_rate)
    print("z_lr_scale\t", z_lr_scale)
    print("str_lr_scale\t", str_lr_scale)
    print("use_three_stage\t", use_three_stage)
    print("warm_epochs\t", warm_epochs)
    print("lambda_tr\t", lambda_tr)
    print("lambda_prior\t", prior_loss_weight)
    print("lambda_str\t", lambda_str)
    print("grad_clip\t", grad_clip)
    print("loss_function\t", loss_function)
    print("metric_method\t", metric_method)
    print("masked_train_val\t", bool(masked_flag))
    if masked_flag:
        print("missing_value\t", missing_value)
    if loss_function.strip().lower() in ("mse_mae", "mse+mae", "mixed"):
        print("mse_weight\t", mse_weight)
    if loss_function.strip().lower() in ("huber", "smooth_l1", "smoothl1"):
        print("huber_beta\t", huber_beta)
    print("lr_scheduler\t", lr_scheduler_name)
    if lr_scheduler_name == "cosine":
        print("cosine_t_max\t", cosine_t_max if cosine_t_max is not None else (epochs - start_epoch))
        print("cosine_eta_min\t", cosine_eta_min)
    print("batch_size\t", batch_size)
    print("epochs\t", epochs)
    print("start_epoch\t", start_epoch)
    print("patience\t", patience if patience > 0 else "disabled")
    print("eval_test_every\t", eval_test_every)

    # 验证仍用配置的 criterion；训练默认按式(273)走 JointForecastLoss(MAE)
    # metric_method=mask 时：验证/训练均忽略 missing_value（METR/BAY 常用 0）
    criterion = build_criterion(
        loss_function,
        DEVICE,
        mse_weight=mse_weight,
        huber_beta=huber_beta,
        masked=bool(masked_flag),
    )
    joint_criterion = JointForecastLoss(
        lambda_tr=lambda_tr,
        lambda_prior=prior_loss_weight,
        lambda_str=lambda_str,
        null_val=train_null_val,
    )

    train_stage = "joint"
    if use_three_stage and warm_epochs > 0:
        apply_stage2_freeze(net)
        train_stage = "warm"
        print("[Stage-2] freeze Θ_str / A_0^{prior}, optimize L_warm = L_pred + λ_tr L_tr")
        optimizer = build_stage_optimizer(
            net, learning_rate, learning_rate * str_lr_scale, stage="warm"
        )
    else:
        apply_stage3_unfreeze(net)
        # 兼容旧差分学习率：结构模块用 str_lr_scale
        optimizer = build_stage_optimizer(
            net, learning_rate, learning_rate * str_lr_scale, stage="joint"
        )
        print("[Stage-3/joint] L_joint = L_pred + λ_tr L_tr + λ_prior L_prior (+ λ_str L_pre)")

    scheduler = None
    if lr_scheduler_name in ("cosine", "cosineannealing", "cos"):
        t_max = cosine_t_max if cosine_t_max is not None else max(1, epochs - start_epoch)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=cosine_eta_min,
        )
    elif lr_scheduler_name not in ("none", "off", "constant", ""):
        raise ValueError("未知 lr_scheduler=%r，可选: none / cosine" % (lr_scheduler_name,))

    sw = SummaryWriter(logdir=params_path, flush_secs=5)
    print(net)

    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf
    early_stop_counter = 0
    start_time = time()

    if start_epoch > 0:
        params_filename = os.path.join(params_path, "epoch_%s.params" % start_epoch)
        net.load_state_dict(torch.load(params_filename, map_location=DEVICE))
        print("load weight from: ", params_filename)

    def _maybe_structure_pretrain_loss():
        """联合阶段可选 L_pre（需 joint_finetune_mvgae 且 λ_str>0）。"""
        if lambda_str <= 0.0 or getattr(net, "mvgae", None) is None:
            return None
        from torch_geometric.utils import negative_sampling

        model = net.mvgae
        x = net.mvgae_x
        eis = net._mvgae_edge_indices()
        ews = net._mvgae_edge_weights()
        z = model.encode(x, eis, ews)
        pos = eis[0]
        if pos.numel() == 0:
            return model.kl_loss()
        neg = negative_sampling(
            edge_index=pos,
            num_nodes=x.size(0),
            num_neg_samples=max(1, pos.size(1)),
        )
        pre_cfg = mvgae_pretrain_config_from_parser(config)
        loss_recon, _, _ = model.structure_recon_loss(
            z,
            pos,
            neg,
            eis,
            ews,
            num_nodes=x.size(0),
            lambda_neg=pre_cfg.lambda_neg,
            lambda_mv=pre_cfg.lambda_mv,
            mv_subsample=pre_cfg.mv_recon_samples,
        )
        beta = float(pre_cfg.kl_weight) if pre_cfg.variational else 0.0
        return loss_recon + beta * model.kl_loss()

    for epoch in range(start_epoch, epochs):
        epoch_start = time()
        params_filename = os.path.join(params_path, "epoch_%s.params" % epoch)

        if (
            use_three_stage
            and train_stage == "warm"
            and epoch >= warm_epochs
        ):
            apply_stage3_unfreeze(net)
            train_stage = "joint"
            optimizer = build_stage_optimizer(
                net, learning_rate, learning_rate * str_lr_scale, stage="joint"
            )
            if scheduler is not None and lr_scheduler_name in ("cosine", "cosineannealing", "cos"):
                remain = max(1, epochs - epoch)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=remain, eta_min=cosine_eta_min
                )
            early_stop_counter = 0
            print(
                "[Stage-3] unfreeze structure | ξ_str/ξ_pred=%.3f | L_joint"
                % str_lr_scale
            )

        if eval_test_every <= 0:
            run_test_eval = epoch == epochs - 1
        else:
            run_test_eval = epoch == epochs - 1 or epoch % eval_test_every == 0
        if run_test_eval:
            evaluate_on_test_mstgcn(net, test_loader, test_target_tensor, sw, epoch, _mean, _std)
        val_loss = compute_val_loss_mstgcn(net, val_loader, criterion, masked_flag, missing_value, sw, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            early_stop_counter = 0
            torch.save(net.state_dict(), params_filename)
            torch.save(net.state_dict(), os.path.join(params_path, "best.params"))
            print("save parameters to file: %s" % params_filename)
        elif patience > 0:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(
                    "early stopping at epoch %d (no val improvement for %d epochs)"
                    % (epoch + 1, patience)
                )
                break

        net.train()
        train_loss_sum = 0.0
        train_batches = 0
        warm_only = train_stage == "warm"
        for batch_index, batch_data in enumerate(train_loader):
            encoder_inputs, labels = batch_data
            optimizer.zero_grad()
            if getattr(net, "temporal_forecaster", None) is not None:
                outputs, aux = net(encoder_inputs, return_aux=True)
                h_tr = aux.get("h_tr")
            else:
                outputs = net(encoder_inputs)
                h_tr = None

            loss_prior = None
            loss_str = None
            if (
                not warm_only
                and getattr(net, "structure_prior", None) is not None
                and prior_loss_weight > 0.0
            ):
                loss_prior = net.structure_prior.prior_preservation_loss()
            if not warm_only and lambda_str > 0.0:
                loss_str = _maybe_structure_pretrain_loss()

            # 式(273)(298)(286)：训练侧统一走 MAE 联合目标
            loss = joint_criterion(
                outputs,
                labels,
                h_tr=h_tr,
                loss_prior=loss_prior,
                loss_str=loss_str,
                warm_only=warm_only,
            )
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)
            optimizer.step()

            global_step += 1
            train_loss_sum += loss.item()
            train_batches += 1
            sw.add_scalar("training_loss", loss.item(), global_step)
            sw.add_scalar("loss_pred", joint_criterion.last["pred"], global_step)
            sw.add_scalar("loss_tr", joint_criterion.last["tr"], global_step)
            sw.add_scalar("train_stage", 2 if warm_only else 3, global_step)
            if loss_prior is not None:
                sw.add_scalar("loss_prior", float(loss_prior.detach()), global_step)
            if loss_str is not None:
                sw.add_scalar("loss_str", float(loss_str.detach()), global_step)
            if isinstance(criterion, CombinedMSEMAELoss):
                sw.add_scalar("loss_mse_raw", criterion.last_mse, global_step)
                sw.add_scalar("loss_mae_raw", criterion.last_mae, global_step)
            if global_step % log_every == 0:
                print(
                    "global step: %s, stage=%s total=%.4f pred=%.4f tr=%.4f prior=%.4f str=%.4f | time=%.2fs"
                    % (
                        global_step,
                        train_stage,
                        joint_criterion.last["total"],
                        joint_criterion.last["pred"],
                        joint_criterion.last["tr"],
                        joint_criterion.last["prior"],
                        joint_criterion.last["str"],
                        time() - start_time,
                    )
                )

        train_loss = train_loss_sum / max(train_batches, 1)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            "[epoch %d/%d] stage=%s train_loss=%.4f val_loss=%.4f lr=%.6g best_epoch=%d patience=%d/%d elapsed=%.1fs"
            % (
                epoch + 1,
                epochs,
                train_stage,
                train_loss,
                val_loss,
                current_lr,
                best_epoch,
                early_stop_counter,
                patience if patience > 0 else 0,
                time() - epoch_start,
            )
        )
        sw.add_scalar("learning_rate", current_lr, epoch)
        if scheduler is not None:
            scheduler.step()

    print("best epoch:", best_epoch)
    print("best val loss:", best_val_loss)
    metrics = predict_main(best_epoch, test_loader, test_target_tensor, _mean, _std, "test")
    if not isinstance(metrics, dict) or "overall" not in metrics:
        print(
            "warning: predict_and_save_results_mstgcn 未返回指标字典 "
            "(服务器上的 lib/utils.py 可能未同步)。"
            "训练与测试评估已完成，跳过 results_summary 写入。"
        )
        print("请同步最新 lib/utils.py 后重跑评估，或从日志读取 all MAE/RMSE/MAPE。")
        return

    overall = metrics["overall"]
    summary = {
        "run_tag": run_tag,
        "params_path": os.path.abspath(params_path),
        "config_path": os.path.abspath(args.config),
        "model_name": model_name,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "epochs_ran": best_epoch + 1 if start_epoch == 0 else epochs,
        "early_stopped": early_stop_counter >= patience if patience > 0 else False,
        "test_metrics": metrics,
        "z_init_sim_to_r100": z_init_info.get("z_init_sim_to_r100"),
        "z_init_similarity_baseline": z_init_info.get("z_init_similarity_baseline"),
    }
    save_results_summary(params_path, summary)
    csv_path = append_runs_summary(
        dataset_name,
        {
            "run_tag": run_tag,
            "params_path": params_path,
            "config_path": args.config,
            "model_name": model_name,
            "seed": seed,
            "learning_rate": learning_rate,
            "z_lr_scale": z_lr_scale,
            "loss_function": loss_function,
            "lr_scheduler": lr_scheduler_name,
            "epochs": epochs,
            "best_epoch": best_epoch,
            "best_val_loss": f"{best_val_loss:.6f}",
            "test_mae": f"{overall['mae']:.4f}",
            "test_rmse": f"{overall['rmse']:.4f}",
            "test_mape": f"{overall['mape']:.4f}",
            "z_init_sim_to_r100": (
                f"{z_init_info['z_init_sim_to_r100']:.6f}"
                if z_init_info.get("z_init_sim_to_r100") is not None
                else ""
            ),
            "use_mvgae_pretrain": z_init_info["use_mvgae_pretrain"],
            "use_node_embed": z_init_info["use_node_embed"],
            "z_init_mode": z_init_info["z_init_mode"],
            "period_split": period_split,
            "fusion_mode": fusion_mode,
            "use_local_branch": use_local_branch,
            "use_global_branch": use_global_branch,
            "joint_finetune_mvgae": joint_finetune_mvgae,
            "end_to_end_mvgae": end_to_end_mvgae,
            "transfer_encoder": transfer_encoder,
            "source_mvgae_checkpoint": source_mvgae_checkpoint,
            "len_input": len_input,
            "num_of_weeks": num_of_weeks,
            "num_of_days": num_of_days,
            "num_of_hours": num_of_hours,
        },
    )
    print("saved results summary:", os.path.join(params_path, "results_summary.json"))
    print("appended run record:", csv_path)


def predict_main(global_step, data_loader, data_target_tensor, _mean, _std, type):
    params_filename = os.path.join(params_path, "epoch_%s.params" % global_step)
    print("load weight from:", params_filename)
    net.load_state_dict(torch.load(params_filename, map_location=DEVICE))
    return predict_and_save_results_mstgcn(
        net,
        data_loader,
        data_target_tensor,
        global_step,
        metric_method,
        _mean,
        _std,
        params_path,
        type,
    )


if __name__ == "__main__":
    train_main()
