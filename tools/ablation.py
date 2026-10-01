#!/usr/bin/env python
# coding: utf-8
"""
PEMS08 消融实验：从 tune 基座生成 configurations/ablation/*.conf 并批量训练。

用法：
  # 仅生成配置
  python tools/run_ablation.py --generate-only

  # 跑主消融表（论文模块有无；骨干随开关变化）
  python tools/run_ablation.py --group main

  # 公平对照：固定 GWN+Transformer，只改节点嵌入 Z
  python tools/run_ablation.py --group fair

  # 跑多视图 / 损失附录 / 全部
  python tools/run_ablation.py --group view
  python tools/run_ablation.py --group loss
  python tools/run_ablation.py --group all

  # 干跑
  python tools/run_ablation.py --group fair --dry-run

前置：建议已有 data/PEMS08/z_init_tune.npy 与 mvgae_pretrain_tune.pt。
结构侧消融（view/single_gae 等）会写独立 z_init，并设 auto_pretrain=True。

注意：wo_embed 会关掉 Dual-Scale / StructTemp 并打开全局 Transformer，
与 Full 不是「只去掉 Z」的公平对比；请优先看 --group fair。
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ABLATION_DIR = ROOT / "configurations" / "ablation"
TRAIN_SCRIPT = ROOT / "train_hybrid.py"
DEFAULT_BASE = ROOT / "configurations" / "PEMS08_multi_period_tune.conf"

# 复用 tune 结构文件的消融（不重训 MVGAE）
TUNE_Z = "./data/PEMS08/z_init_tune.npy"
TUNE_CKPT = "./data/PEMS08/mvgae_pretrain_tune.pt"

# 固定 Hybrid 骨干（GWN 局部 + Transformer 全局），用于公平嵌入消融
_HYBRID_BACKBONE = {
    "use_dual_scale_spatial": "False",
    "use_structure_aware_temporal": "False",
    "use_local_branch": "True",
    "use_global_branch": "True",
    "use_abase_prior": "False",
    "prior_loss_weight": "0.0",
    "lambda_tr": "0.0",
}

# name -> (group, description, patches)
# patches: dict of key -> new value (匹配行首 key = ...)
ABLATIONS: dict[str, tuple[str, str, dict]] = {
    # ----- Table 1: component -----
    "full": (
        "main",
        "完整模型（tune 基座：DualScale + StructTemp）",
        {
            "model_name": "ablate_pems08_full",
            "auto_pretrain": "False",
            "z_init_filename": TUNE_Z,
            "mvgae_checkpoint_filename": TUNE_CKPT,
        },
    ),
    "wo_dual": (
        "main",
        "w/o Dual-Scale Spatial（回退 GWN + 结构时序）",
        {
            "model_name": "ablate_pems08_wo_dual",
            "auto_pretrain": "False",
            "z_init_filename": TUNE_Z,
            "mvgae_checkpoint_filename": TUNE_CKPT,
            "use_dual_scale_spatial": "False",
            "use_structure_aware_temporal": "True",
            "use_global_branch": "False",
        },
    ),
    "wo_tem": (
        "main",
        "w/o Structure-Aware Temporal（双尺度 + Transformer）",
        {
            "model_name": "ablate_pems08_wo_tem",
            "auto_pretrain": "False",
            "z_init_filename": TUNE_Z,
            "mvgae_checkpoint_filename": TUNE_CKPT,
            "use_dual_scale_spatial": "True",
            "use_structure_aware_temporal": "False",
            "use_global_branch": "True",
        },
    ),
    "wo_mvgae": (
        "main",
        "论文路径上 w/o MVGAE（随机 Z，仍 DualScale+StructTemp）",
        {
            "model_name": "ablate_pems08_wo_mvgae",
            "auto_pretrain": "False",
            "use_mvgae_pretrain": "False",
            "use_node_embed": "True",
            "z_init_filename": "./data/PEMS08/z_init_ablate_random.npy",
            "mvgae_checkpoint_filename": TUNE_CKPT,
        },
    ),
    "wo_prior": (
        "main",
        "w/o A^{prior} 结构先验（phy-only 锚点）",
        {
            "model_name": "ablate_pems08_wo_prior",
            "auto_pretrain": "False",
            "z_init_filename": TUNE_Z,
            "mvgae_checkpoint_filename": TUNE_CKPT,
            "use_abase_prior": "False",
            "prior_loss_weight": "0.0",
        },
    ),
    "seq_dual_tem": (
        "main",
        "无结构嵌入 + DualScale + StructTemp（纯序列论文路径）",
        {
            "model_name": "ablate_pems08_seq_dual_tem",
            "auto_pretrain": "False",
            "use_mvgae_pretrain": "False",
            "use_node_embed": "False",
            # 保留 tune 的 DualScale + StructTemp；Z 路径仅用于加载 A_prior 侧车
            "use_dual_scale_spatial": "True",
            "use_structure_aware_temporal": "True",
            "use_global_branch": "True",  # 会被 StructTemp 强制关掉
            "use_abase_prior": "True",
            "z_init_filename": TUNE_Z,
            "mvgae_checkpoint_filename": TUNE_CKPT,
        },
    ),
    # ----- Fair: 固定 GWN+Transformer，只改 Z -----
    "hybrid_with_z": (
        "fair",
        "Hybrid骨干 + MVGAE预训练Z（公平对照）",
        {
            "model_name": "ablate_pems08_hybrid_with_z",
            "auto_pretrain": "False",
            "use_mvgae_pretrain": "True",
            "use_node_embed": "True",
            "z_init_filename": TUNE_Z,
            "mvgae_checkpoint_filename": TUNE_CKPT,
            **_HYBRID_BACKBONE,
        },
    ),
    "hybrid_rand_z": (
        "fair",
        "Hybrid骨干 + 随机可学习Z（公平对照）",
        {
            "model_name": "ablate_pems08_hybrid_rand_z",
            "auto_pretrain": "False",
            "use_mvgae_pretrain": "False",
            "use_node_embed": "True",
            "z_init_filename": "./data/PEMS08/z_init_ablate_hybrid_rand.npy",
            "mvgae_checkpoint_filename": TUNE_CKPT,
            **_HYBRID_BACKBONE,
        },
    ),
    "wo_embed": (
        "fair",
        "Hybrid骨干 + 无节点嵌入（即原 wo_embed；勿与 Full 直接比）",
        {
            "model_name": "ablate_pems08_wo_embed",
            "auto_pretrain": "False",
            "use_mvgae_pretrain": "False",
            "use_node_embed": "False",
            "z_init_filename": TUNE_Z,
            "mvgae_checkpoint_filename": TUNE_CKPT,
            **_HYBRID_BACKBONE,
        },
    ),
    # ----- Table 2: multi-view（需重训结构） -----
    "single_view": (
        "view",
        "仅物理视图 phy（num_heads=1）",
        {
            "model_name": "ablate_pems08_single_view",
            "auto_pretrain": "True",
            "num_heads": "1",
            "view_names": "phy",
            "z_init_filename": "./data/PEMS08/z_init_ablate_single_view.npy",
            "mvgae_checkpoint_filename": "./data/PEMS08/mvgae_ablate_single_view.pt",
        },
    ),
    "single_gae": (
        "view",
        "单视图确定性 GAE",
        {
            "model_name": "ablate_pems08_single_gae",
            "auto_pretrain": "True",
            "num_heads": "1",
            "view_names": "phy",
            "variational": "False",
            "kl_weight": "0.0",
            "z_init_filename": "./data/PEMS08/z_init_ablate_single_gae.npy",
            "mvgae_checkpoint_filename": "./data/PEMS08/mvgae_ablate_single_gae.pt",
        },
    ),
    "wo_view_2nd": (
        "view",
        "w/o A^{2nd}",
        {
            "model_name": "ablate_pems08_wo_view_2nd",
            "auto_pretrain": "True",
            "view_names": "phy,role,dir",
            "num_heads": "3",
            "z_init_filename": "./data/PEMS08/z_init_ablate_wo_2nd.npy",
            "mvgae_checkpoint_filename": "./data/PEMS08/mvgae_ablate_wo_2nd.pt",
        },
    ),
    "wo_view_role": (
        "view",
        "w/o A^{role}",
        {
            "model_name": "ablate_pems08_wo_view_role",
            "auto_pretrain": "True",
            "view_names": "phy,2nd,dir",
            "num_heads": "3",
            "z_init_filename": "./data/PEMS08/z_init_ablate_wo_role.npy",
            "mvgae_checkpoint_filename": "./data/PEMS08/mvgae_ablate_wo_role.pt",
        },
    ),
    "wo_view_dir": (
        "view",
        "w/o A^{dir}",
        {
            "model_name": "ablate_pems08_wo_view_dir",
            "auto_pretrain": "True",
            "view_names": "phy,2nd,role",
            "num_heads": "3",
            "z_init_filename": "./data/PEMS08/z_init_ablate_wo_dir.npy",
            "mvgae_checkpoint_filename": "./data/PEMS08/mvgae_ablate_wo_dir.pt",
        },
    ),
    "view_fusion_mean": (
        "view",
        "视图融合 mean（w/o 注意力）",
        {
            "model_name": "ablate_pems08_view_fusion_mean",
            "auto_pretrain": "True",
            "view_fusion": "mean",
            "z_init_filename": "./data/PEMS08/z_init_ablate_fusion_mean.npy",
            "mvgae_checkpoint_filename": "./data/PEMS08/mvgae_ablate_fusion_mean.pt",
        },
    ),
    # ----- Table 3: loss / training -----
    "wo_l_mv": (
        "loss",
        "w/o L_mv（lambda_mv=0）",
        {
            "model_name": "ablate_pems08_wo_l_mv",
            "auto_pretrain": "True",
            "lambda_mv": "0.0",
            "z_init_filename": "./data/PEMS08/z_init_ablate_wo_lmv.npy",
            "mvgae_checkpoint_filename": "./data/PEMS08/mvgae_ablate_wo_lmv.pt",
        },
    ),
    "wo_kl": (
        "loss",
        "w/o KL / 确定性后验",
        {
            "model_name": "ablate_pems08_wo_kl",
            "auto_pretrain": "True",
            "variational": "False",
            "kl_weight": "0.0",
            "z_init_filename": "./data/PEMS08/z_init_ablate_wo_kl.npy",
            "mvgae_checkpoint_filename": "./data/PEMS08/mvgae_ablate_wo_kl.pt",
        },
    ),
    "wo_ltr": (
        "loss",
        "w/o L_tr",
        {
            "model_name": "ablate_pems08_wo_ltr",
            "auto_pretrain": "False",
            "z_init_filename": TUNE_Z,
            "mvgae_checkpoint_filename": TUNE_CKPT,
            "lambda_tr": "0.0",
        },
    ),
    "wo_lprior": (
        "loss",
        "w/o L_prior",
        {
            "model_name": "ablate_pems08_wo_lprior",
            "auto_pretrain": "False",
            "z_init_filename": TUNE_Z,
            "mvgae_checkpoint_filename": TUNE_CKPT,
            "prior_loss_weight": "0.0",
        },
    ),
    "wo_period": (
        "loss",
        "w/o 多周期分路（period_split=False）",
        {
            "model_name": "ablate_pems08_wo_period",
            "auto_pretrain": "False",
            "z_init_filename": TUNE_Z,
            "mvgae_checkpoint_filename": TUNE_CKPT,
            "period_split": "False",
        },
    ),
    "joint_ft": (
        "loss",
        "联合微调 MVGAE（对照冻结 Z）",
        {
            "model_name": "ablate_pems08_joint_ft",
            "auto_pretrain": "False",
            "z_init_filename": TUNE_Z,
            "mvgae_checkpoint_filename": TUNE_CKPT,
            "joint_finetune_mvgae": "True",
            "lambda_str": "0.05",
            "str_lr_scale": "0.05",
            "mvgae_lr_scale": "0.05",
        },
    ),
}

EXTRA_KEYS_AFTER = {
    # 插入到 [MVGAE] 段末尾附近（若不存在则追加）
    "view_names": "num_heads",
    "view_fusion": "variational",
    "mvgae_lr_scale": "z_lr_scale",
}


def _set_or_insert(text: str, key: str, value: str, after_key: str | None = None) -> str:
    pattern = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*).*$", re.MULTILINE)
    if pattern.search(text):
        return pattern.sub(rf"\g<1>{value}", text, count=1)
    # 插入：优先跟在 after_key 后
    anchor = after_key or EXTRA_KEYS_AFTER.get(key)
    if anchor:
        anchor_pat = re.compile(rf"^(\s*{re.escape(anchor)}\s*=\s*.*)$", re.MULTILINE)
        m = anchor_pat.search(text)
        if m:
            insert = f"{m.group(1)}\n{key} = {value}"
            return text[: m.start()] + insert + text[m.end() :]
    # 否则放在 [MVGAE] 段首之后
    section = re.search(r"^\[MVGAE\]\s*$", text, re.MULTILINE)
    if section and key in ("view_names", "view_fusion", "num_heads", "variational", "lambda_mv", "kl_weight"):
        pos = section.end()
        return text[:pos] + f"\n{key} = {value}" + text[pos:]
    section_h = re.search(r"^\[Hybrid\]\s*$", text, re.MULTILINE)
    if section_h and key in ("mvgae_lr_scale",):
        pos = section_h.end()
        return text[:pos] + f"\n{key} = {value}" + text[pos:]
    return text.rstrip() + f"\n{key} = {value}\n"


def patch_config(base_text: str, patches: dict, header: str) -> str:
    text = base_text
    # 去掉原文件顶注释，换成消融说明
    if text.lstrip().startswith(";"):
        lines = text.splitlines()
        while lines and (lines[0].startswith(";") or not lines[0].strip()):
            lines.pop(0)
        text = "\n".join(lines) + "\n"
    text = header + "\n" + text
    for key, value in patches.items():
        text = _set_or_insert(text, key, str(value))
    return text if text.endswith("\n") else text + "\n"


def config_path(name: str) -> Path:
    return ABLATION_DIR / f"PEMS08_{name}.conf"


def generate_all(base_path: Path) -> list[Path]:
    base_text = base_path.read_text(encoding="utf-8")
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    readme_lines = [
        "# PEMS08 Ablation Configs",
        "",
        "由 `tools/run_ablation.py` 从 `PEMS08_multi_period_tune.conf` 生成。",
        "",
        "## 重要：公平对照（`--group fair`）",
        "",
        "`wo_embed` 会退回 **GWN + Temporal Transformer**，与 Full（DualScale + StructTemp）骨干不同。",
        "若要回答「节点嵌入 Z 有没有用」，请对比：",
        "",
        "| 实验 | 含义 |",
        "|------|------|",
        "| `hybrid_with_z` | Hybrid 骨干 + MVGAE 预训练 Z |",
        "| `hybrid_rand_z` | Hybrid 骨干 + 随机可学习 Z |",
        "| `wo_embed` | Hybrid 骨干 + 无节点嵌入 |",
        "",
        "| 文件 | 组 | 说明 |",
        "|------|----|------|",
    ]
    for name, (group, desc, patches) in ABLATIONS.items():
        header = f"; Ablation [{group}] {name}: {desc}"
        content = patch_config(base_text, patches, header)
        path = config_path(name)
        path.write_text(content, encoding="utf-8")
        written.append(path)
        readme_lines.append(f"| `{path.name}` | {group} | {desc} |")
    readme_lines.extend(
        [
            "",
            "## 运行",
            "",
            "```bash",
            "python tools/run_ablation.py --group fair   # 优先：固定骨干只改 Z",
            "python tools/run_ablation.py --group main",
            "python tools/run_ablation.py --group view",
            "python tools/run_ablation.py --group loss",
            "```",
            "",
        ]
    )
    (ABLATION_DIR / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    return written


def names_for_group(group: str) -> list[str]:
    group = group.strip().lower()
    if group == "all":
        return list(ABLATIONS.keys())
    return [n for n, (g, _, _) in ABLATIONS.items() if g == group]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate/run PEMS08 ablation suite")
    parser.add_argument(
        "--base",
        default=str(DEFAULT_BASE.relative_to(ROOT)).replace("\\", "/"),
        help="基座配置（默认 PEMS08 tune）",
    )
    parser.add_argument(
        "--group",
        default="main",
        choices=["main", "fair", "view", "loss", "all"],
        help="消融组（fair=固定 Hybrid 骨干只改 Z）",
    )
    parser.add_argument("--generate-only", action="store_true", help="只生成配置")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令")
    parser.add_argument("--only", type=str, default=None, help="逗号分隔实验名，覆盖 --group")
    parser.add_argument("--python", default=sys.executable, help="Python 解释器")
    parser.add_argument("--list", action="store_true", help="列出可用消融名")
    args = parser.parse_args()

    if args.list:
        for name, (group, desc, _) in ABLATIONS.items():
            print(f"{name:18s} [{group:4s}] {desc}")
        return 0

    base_path = (ROOT / args.base).resolve()
    if not base_path.is_file():
        print(f"Base config not found: {base_path}", file=sys.stderr)
        return 1

    written = generate_all(base_path)
    print(f"Generated {len(written)} configs under {ABLATION_DIR.relative_to(ROOT)}")

    if args.generate_only:
        return 0

    if args.only:
        names = [x.strip() for x in args.only.split(",") if x.strip()]
        bad = [n for n in names if n not in ABLATIONS]
        if bad:
            print(f"Unknown ablation names: {bad}", file=sys.stderr)
            return 1
    else:
        names = names_for_group(args.group)

    if not names:
        print(f"No ablations in group={args.group}", file=sys.stderr)
        return 1

    print(f"Will run {len(names)} experiments: {', '.join(names)}")
    failed: list[str] = []
    for name in names:
        conf = config_path(name)
        rel = conf.relative_to(ROOT).as_posix()
        cmd = [args.python, str(TRAIN_SCRIPT), "--config", rel]
        print("\n==>", " ".join(cmd))
        if args.dry_run:
            continue
        ret = subprocess.call(cmd, cwd=str(ROOT))
        if ret != 0:
            print(f"[FAIL] {name} exit={ret}", file=sys.stderr)
            failed.append(name)

    if args.dry_run:
        return 0
    if failed:
        print(f"\nFailed ({len(failed)}): {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"\nAll {len(names)} ablations finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
