"""
将 LibCity 格式 (*.geo/*.rel/*.dyna) 转为本工程可读的：
  - {name}.npz  (key=data, shape [T,N,1])
  - {name}.csv   (from,to,cost；节点已重映射为 0..N-1)
  - coords.csv   (id,x,y；x=lon,y=lat)

仅依赖标准库 + numpy（不需要 pandas）。
"""
from __future__ import annotations

import argparse
import ast
import csv
import os
from typing import Dict, List, Tuple

import numpy as np


def _parse_coords(raw: str) -> Tuple[float, float]:
    xy = ast.literal_eval(raw)
    return float(xy[0]), float(xy[1])


def _read_geo(geo_path: str) -> Tuple[List[int], np.ndarray]:
    geo_ids: List[int] = []
    coords_list: List[Tuple[float, float]] = []
    with open(geo_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            geo_ids.append(int(row["geo_id"]))
            coords_list.append(_parse_coords(row["coordinates"]))
    coords = np.asarray(coords_list, dtype=np.float32)
    return geo_ids, coords


def _write_coords_csv(path: str, coords: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("id,x,y\n")
        for i in range(coords.shape[0]):
            f.write(f"{i},{coords[i, 0]:.8f},{coords[i, 1]:.8f}\n")


def _write_adj_csv(rel_path: str, out_path: str, id2idx: Dict[int, int]) -> int:
    n_edges = 0
    with open(rel_path, "r", encoding="utf-8", newline="") as fin, open(
        out_path, "w", encoding="utf-8", newline="\n"
    ) as fout:
        reader = csv.DictReader(fin)
        fout.write("from,to,cost\n")
        for row in reader:
            o = int(row["origin_id"])
            d = int(row["destination_id"])
            if o == d:
                continue
            if o not in id2idx or d not in id2idx:
                continue
            cost = float(row["cost"])
            fout.write(f"{id2idx[o]},{id2idx[d]},{cost}\n")
            n_edges += 1
    return n_edges


def _build_signal_npz(
    dyna_path: str, id2idx: Dict[int, int], n: int, npz_path: str
) -> Tuple[int, float, float]:
    """
    两遍扫描 dyna：
      1) 收集有序时间戳
      2) 填充 [T,N,1]
    """
    times = set()
    with open(dyna_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.add(row["time"])
    time_list = sorted(times)
    time2idx = {tm: i for i, tm in enumerate(time_list)}
    t = len(time_list)

    data = np.zeros((t, n, 1), dtype=np.float32)
    filled = np.zeros((t, n), dtype=np.bool_)
    with open(dyna_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gid = int(row["entity_id"])
            if gid not in id2idx:
                raise ValueError(f"dyna 中存在 geo 未覆盖的 entity_id={gid}")
            ti = time2idx[row["time"]]
            ni = id2idx[gid]
            data[ti, ni, 0] = float(row["traffic_speed"])
            filled[ti, ni] = True

    # 缺失保持 0（与常见 METR/BAY 约定一致，可用 metric_method=mask）
    del filled
    np.savez_compressed(npz_path, data=data)
    return t, float(data.mean()), float((data != 0).mean())


def convert_dataset(data_dir: str, name: str) -> None:
    geo_path = os.path.join(data_dir, f"{name}.geo")
    rel_path = os.path.join(data_dir, f"{name}.rel")
    dyna_path = os.path.join(data_dir, f"{name}.dyna")
    for p in (geo_path, rel_path, dyna_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    geo_ids, coords = _read_geo(geo_path)
    id2idx = {gid: i for i, gid in enumerate(geo_ids)}
    n = len(geo_ids)

    coords_path = os.path.join(data_dir, "coords.csv")
    _write_coords_csv(coords_path, coords)
    print(f"[coords] {coords_path} N={n}")

    csv_path = os.path.join(data_dir, f"{name}.csv")
    n_edges = _write_adj_csv(rel_path, csv_path, id2idx)
    print(f"[adj] {csv_path} edges={n_edges}")

    npz_path = os.path.join(data_dir, f"{name}.npz")
    print(f"[dyna] loading {dyna_path} ...")
    t, mean_v, nonzero = _build_signal_npz(dyna_path, id2idx, n, npz_path)
    print(f"[npz] {npz_path} shape=({t}, {n}, 1) mean={mean_v:.4f} nonzero_ratio={nonzero:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("METR_LA", "PEMS_BAY", "both"),
        default="both",
    )
    parser.add_argument("--root", default="./data")
    args = parser.parse_args()
    names = ("METR_LA", "PEMS_BAY") if args.dataset == "both" else (args.dataset,)
    for name in names:
        convert_dataset(os.path.join(args.root, name), name)


if __name__ == "__main__":
    main()
