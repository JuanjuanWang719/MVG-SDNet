"""
真四视图空间关系构图：

1. A^{phy}  — 有向物理道路连接（可不对称；可选式(33)距离加权）
2. A^{2nd}  — 净二阶传播：(A^{phy})^2 ⊙ M^{2nd}，再 (D+εI)^{-1} 行归一化（有向）
3. A^{role} — 拓扑角色相似：出/入度、邻域总度、B^{2nd}、A^u 聚类 → 余弦 Top-K 后平均对称化
4. A^{dir}  — 方向感知结构（角色 × 距离衰减 × 方向一致性，有向 Top-K）

消息传递采用出度行归一化 Â=(D^{out})^{-1}(A+I)，见模型 RowNormConv。
无 GPS 时用边 cost 最短路 + classical MDS 估计 p_i，再由局部边估计方向向量 e_i。
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor

EPS = 1e-8
VIEW_NAMES = ("phy", "2nd", "role", "dir")


@dataclass
class MultiViewGraphConfig:
    hop_order: int = 2
    # A^{2nd} 可选 Top-K；0/None 表示保留全部净二阶边（论文默认）
    view_hop_k: int = 0
    view_role_k: int = 10
    view_dir_k: int = 10
    sigma_dir: Union[str, float] = "median"
    phy_distance_weight: bool = True
    sigma_d: Union[str, float] = "median"
    eps_2nd: float = 1e-8
    seed: int = 42
    coords_filename: Optional[str] = None


def load_id_dict(id_filename: Optional[str]) -> Optional[dict]:
    if not id_filename:
        return None
    with open(id_filename, "r", encoding="utf-8") as f:
        return {int(i): idx for idx, i in enumerate(f.read().strip().split("\n"))}


def _resolve_sigma(values: np.ndarray, sigma: Union[str, float]) -> float:
    if isinstance(sigma, str) and sigma.strip().lower() == "median":
        v = values[np.isfinite(values) & (values > 0)]
        return float(np.median(v)) if v.size > 0 else 1.0
    return max(float(sigma), EPS)


def load_directed_edge_list(
    distance_df_filename: str,
    num_of_vertices: int,
    id_filename: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    读取有向边列表 (src, dst, cost)，不对称化。
    同一 (i,j) 多条记录取最小 cost。
    """
    id_dict = load_id_dict(id_filename)
    best: dict[tuple[int, int], float] = {}
    with open(distance_df_filename, "r", encoding="utf-8") as f:
        f.readline()
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            i, j = int(row[0]), int(row[1])
            cost = float(row[2]) if len(row) >= 3 else 1.0
            if id_dict is not None:
                if i not in id_dict or j not in id_dict:
                    continue
                i, j = id_dict[i], id_dict[j]
            if not (0 <= i < num_of_vertices and 0 <= j < num_of_vertices):
                continue
            if i == j:
                continue
            key = (i, j)
            if key not in best or cost < best[key]:
                best[key] = cost
    if not best:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )
    src = np.fromiter((k[0] for k in best.keys()), dtype=np.int64, count=len(best))
    dst = np.fromiter((k[1] for k in best.keys()), dtype=np.int64, count=len(best))
    cost = np.fromiter(best.values(), dtype=np.float32, count=len(best))
    return src, dst, cost


def build_phy_adjacency(
    distance_df_filename: str,
    num_of_vertices: int,
    id_filename: Optional[str] = None,
    distance_weight: bool = True,
    sigma_d: Union[str, float] = "median",
) -> np.ndarray:
    """
    构造有向物理视图 A^{phy}（不对称为对称）。
    - distance_weight=False：二值 A_ij=1 iff (i,j)∈E
    - distance_weight=True：式(33) A_ij = exp(-d_ij^2 / σ_d^2)
    """
    src, dst, cost = load_directed_edge_list(
        distance_df_filename, num_of_vertices, id_filename
    )
    adj = np.zeros((num_of_vertices, num_of_vertices), dtype=np.float32)
    if src.size == 0:
        return adj
    if distance_weight:
        sigma = _resolve_sigma(cost.astype(np.float64), sigma_d)
        weights = np.exp(-(cost.astype(np.float64) ** 2) / (sigma ** 2)).astype(np.float32)
    else:
        weights = np.ones_like(cost, dtype=np.float32)
    adj[src, dst] = weights
    np.fill_diagonal(adj, 0.0)
    return adj


def load_distance_adjacency(
    distance_df_filename: str,
    num_of_vertices: int,
    id_filename: Optional[str] = None,
) -> np.ndarray:
    """兼容旧接口：默认返回有向二值 A^{phy}（不对称化）。"""
    return build_phy_adjacency(
        distance_df_filename,
        num_of_vertices,
        id_filename=id_filename,
        distance_weight=False,
        sigma_d="median",
    )


def load_distance_cost_matrix(
    distance_df_filename: str,
    num_of_vertices: int,
    id_filename: Optional[str] = None,
) -> np.ndarray:
    """加权对称距离矩阵（缺失为 inf）。"""
    dist = np.full((num_of_vertices, num_of_vertices), np.inf, dtype=np.float64)
    np.fill_diagonal(dist, 0.0)
    id_dict = load_id_dict(id_filename)
    with open(distance_df_filename, "r", encoding="utf-8") as f:
        f.readline()
        for row in csv.reader(f):
            if len(row) != 3:
                continue
            i, j = int(row[0]), int(row[1])
            cost = float(row[2])
            if id_dict is not None:
                i, j = id_dict[i], id_dict[j]
            if 0 <= i < num_of_vertices and 0 <= j < num_of_vertices:
                if cost < dist[i, j]:
                    dist[i, j] = cost
                    dist[j, i] = cost
    return dist


def floyd_warshall(dist: np.ndarray) -> np.ndarray:
    """全源最短路（原地式拷贝）。"""
    d = dist.copy()
    n = d.shape[0]
    for k in range(n):
        d = np.minimum(d, d[:, k : k + 1] + d[k : k + 1, :])
    return d


def classical_mds(dist: np.ndarray, n_components: int = 2, seed: int = 42) -> np.ndarray:
    """
    Classical MDS：由距离矩阵嵌入到 R^{n_components}。
    不可达距离用有限大值替换；结果对符号/旋转不唯一，固定 seed 做符号翻转稳定化。
    """
    n = dist.shape[0]
    finite = np.isfinite(dist)
    if not finite.all():
        max_fin = float(np.max(dist[finite])) if finite.any() else 1.0
        fill = max_fin * 2.0 + 1.0
        d = np.where(finite, dist, fill).astype(np.float64)
    else:
        d = dist.astype(np.float64)

    d2 = d ** 2
    h = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * h @ d2 @ h
    evals, evecs = np.linalg.eigh(b)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    pos_mask = evals > 1e-10
    k = min(n_components, int(pos_mask.sum()))
    if k == 0:
        rng = np.random.default_rng(seed)
        return rng.normal(size=(n, n_components)).astype(np.float32)

    coords = evecs[:, :k] * np.sqrt(evals[:k])
    if k < n_components:
        pad = np.zeros((n, n_components - k), dtype=np.float64)
        coords = np.concatenate([coords, pad], axis=1)

    # 符号稳定：使第一主成分与正半轴相关
    rng = np.random.default_rng(seed)
    for c in range(n_components):
        if coords[:, c].sum() < 0:
            coords[:, c] *= -1.0
        if abs(coords[:, c].sum()) < 1e-12 and rng.random() < 0.5:
            coords[:, c] *= -1.0
    return coords.astype(np.float32)


def load_coords_file(
    coords_filename: str,
    num_of_vertices: int,
    id_filename: Optional[str] = None,
) -> np.ndarray:
    """coords CSV: id,x,y 或 x,y（按行序）。"""
    id_dict = load_id_dict(id_filename)
    coords = np.zeros((num_of_vertices, 2), dtype=np.float32)
    filled = np.zeros(num_of_vertices, dtype=bool)
    with open(coords_filename, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"空坐标文件: {coords_filename}")
    start = 1 if rows[0] and rows[0][0].lower() in ("id", "from", "node", "index") else 0
    row_idx = 0
    for row in rows[start:]:
        if len(row) >= 3:
            nid, x, y = row[0], float(row[1]), float(row[2])
            try:
                i = id_dict[int(nid)] if id_dict is not None else int(nid)
            except (KeyError, ValueError):
                continue
            if 0 <= i < num_of_vertices:
                coords[i] = (x, y)
                filled[i] = True
        elif len(row) == 2:
            if row_idx < num_of_vertices:
                coords[row_idx] = (float(row[0]), float(row[1]))
                filled[row_idx] = True
                row_idx += 1
    if not filled.all():
        missing = int((~filled).sum())
        raise ValueError(f"坐标文件缺少 {missing} 个节点: {coords_filename}")
    return coords


def estimate_node_positions(
    adj_filename: str,
    num_of_vertices: int,
    id_filename: Optional[str] = None,
    coords_filename: Optional[str] = None,
    seed: int = 42,
) -> np.ndarray:
    if coords_filename and os.path.isfile(coords_filename):
        return load_coords_file(coords_filename, num_of_vertices, id_filename)
    cost = load_distance_cost_matrix(adj_filename, num_of_vertices, id_filename)
    sp = floyd_warshall(cost)
    return classical_mds(sp, n_components=2, seed=seed)


def estimate_direction_vectors(positions: np.ndarray, adj_phy: np.ndarray) -> np.ndarray:
    """
    对每个节点，用物理邻居在嵌入平面上的位移加权平均并单位化，得 e_i。
    度为 0 时返回零向量。
    """
    n = positions.shape[0]
    e = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        nbrs = np.where(adj_phy[i] > 0)[0]
        if nbrs.size == 0:
            continue
        disp = positions[nbrs] - positions[i]
        norms = np.linalg.norm(disp, axis=1, keepdims=True)
        unit = disp / (norms + EPS)
        vec = unit.mean(axis=0)
        nrm = np.linalg.norm(vec)
        if nrm > EPS:
            e[i] = vec / nrm
    return e.astype(np.float32)


def build_role_features(
    adj_phy: np.ndarray,
    b_2nd: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    式(45)–(55)：拓扑角色向量
      r_i = [d_out, d_in, bar_d, s^{2nd}, C]，再按维标准化。
    """
    a = adj_phy.astype(np.float64).copy()
    np.fill_diagonal(a, 0.0)
    n = a.shape[0]

    # (45)(46)(47)：出度 / 入度 / 总连接度（二值连通）
    d_out = (a > 0).sum(axis=1).astype(np.float64)
    d_in = (a > 0).sum(axis=0).astype(np.float64)
    d_tot = d_out + d_in

    # (48)(49)：无向支撑图与邻域
    a_u = ((a + a.T) > 0).astype(np.float64)
    np.fill_diagonal(a_u, 0.0)
    d_u = a_u.sum(axis=1)

    # (50)：平均邻居总度；孤立节点为 0
    neighbor_tot = a_u @ d_tot
    bar_d = np.zeros(n, dtype=np.float64)
    mask_n = d_u > 0
    bar_d[mask_n] = neighbor_tot[mask_n] / d_u[mask_n]

    # (51)：二阶传播强度
    if b_2nd is None:
        b_2nd = compute_net_second_order(adj_phy)
    s_2nd = b_2nd.astype(np.float64).sum(axis=1)

    # (52)(53)：基于 A^u 的局部聚类系数
    a3_diag = np.diag(a_u @ a_u @ a_u)
    clustering = np.zeros(n, dtype=np.float64)
    mask_c = d_u >= 2
    clustering[mask_c] = a3_diag[mask_c] / (d_u[mask_c] * (d_u[mask_c] - 1.0))

    # (54)(55)
    feats = np.stack([d_out, d_in, bar_d, s_2nd, clustering], axis=1)
    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True)
    feats = (feats - mean) / (std + float(eps))
    return feats.astype(np.float32)


def role_similarity_matrix(role_features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """式(56)：标准化角色向量的余弦相似度，对角为 0。"""
    norms = np.linalg.norm(role_features.astype(np.float64), axis=1, keepdims=True)
    denom = norms @ norms.T + float(eps)
    s = (role_features.astype(np.float64) @ role_features.astype(np.float64).T) / denom
    np.fill_diagonal(s, 0.0)
    return s.astype(np.float32)


def row_topk_sparsify(scores: np.ndarray, k: int, symmetric: bool = False) -> np.ndarray:
    """按行保留 Top-K 正分；可选事后用 max 对称化（角色视图请改用平均对称化）。"""
    n = scores.shape[0]
    k = max(1, min(int(k), n - 1))
    out = np.zeros_like(scores, dtype=np.float32)
    for i in range(n):
        row = scores[i].copy()
        row[i] = -np.inf
        if not np.isfinite(row).any() or np.nanmax(row) <= -np.inf:
            continue
        idx = np.argpartition(row, -k)[-k:]
        idx = idx[np.isfinite(row[idx]) & (row[idx] > -np.inf)]
        if idx.size == 0:
            continue
        pos = idx[row[idx] > 0]
        if pos.size == 0:
            continue
        out[i, pos] = row[pos].astype(np.float32)
    if symmetric:
        out = np.maximum(out, out.T)
    return out


def compute_net_second_order(
    adj_phy: np.ndarray,
    hop_order: int = 2,
) -> np.ndarray:
    """B^{2nd} = (A^{phy})^h ⊙ M^{2nd}（未行归一化）。"""
    if hop_order < 2:
        raise ValueError("hop_order 至少为 2")
    a = adj_phy.astype(np.float64).copy()
    np.fill_diagonal(a, 0.0)
    power = a.copy()
    for _ in range(hop_order - 1):
        power = power @ a
    m = (a == 0.0).astype(np.float64)
    np.fill_diagonal(m, 0.0)
    b = power * m
    np.fill_diagonal(b, 0.0)
    return b


def build_2nd_adjacency(
    adj_phy: np.ndarray,
    hop_order: int = 2,
    top_k: Optional[int] = None,
    eps: float = 1e-8,
    b_2nd: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    按式(38)–(43)构造有向 A^{2nd}：
      B = (A^{phy})^2 ⊙ M^{2nd},
      A^{2nd} = (D^{2nd} + εI)^{-1} B.
    """
    b = (
        b_2nd.astype(np.float64)
        if b_2nd is not None
        else compute_net_second_order(adj_phy, hop_order=hop_order)
    )

    if top_k is not None and int(top_k) > 0:
        b = row_topk_sparsify(b.astype(np.float32), int(top_k), symmetric=False).astype(
            np.float64
        )

    row_sum = b.sum(axis=1)
    d_inv = 1.0 / (row_sum + float(eps))
    return (d_inv[:, None] * b).astype(np.float32)


# 兼容旧名
build_hop_adjacency = build_2nd_adjacency


def build_role_adjacency(s_role: np.ndarray, top_k: int = 10) -> np.ndarray:
    """
    式(57)(58)：行 Top-K_r 且 max(0,S)，再 (Â + Â^T)/2 对称化。
    """
    hat = row_topk_sparsify(s_role, top_k, symmetric=False)
    a_role = 0.5 * (hat + hat.T)
    np.fill_diagonal(a_role, 0.0)
    return a_role.astype(np.float32)


def build_dir_adjacency(
    positions: np.ndarray,
    direction_vectors: np.ndarray,
    s_role: np.ndarray,
    top_k: int = 10,
    sigma_dir: Union[str, float] = "median",
) -> np.ndarray:
    """
    S_ij^dir = max(0, S_ij^role) * ψ_ij^dist * κ_ij
    κ = κ_src * κ_dst * κ_ori（有向、非对称）。
    """
    n = positions.shape[0]
    p = positions.astype(np.float64)
    e = direction_vectors.astype(np.float64)

    diff = p[None, :, :] - p[:, None, :]  # [i,j,:] = p_j - p_i
    dist = np.linalg.norm(diff, axis=-1)
    delta = diff / (dist[..., None] + EPS)

    kappa_src = np.maximum(0.0, np.einsum("id,ijd->ij", e, delta))
    kappa_dst = np.maximum(0.0, np.einsum("jd,ijd->ij", e, delta))
    kappa_ori = (1.0 + (e @ e.T)) * 0.5
    kappa = kappa_src * kappa_dst * kappa_ori

    if isinstance(sigma_dir, str) and sigma_dir.strip().lower() == "median":
        tri = dist[np.triu_indices(n, k=1)]
        tri = tri[np.isfinite(tri) & (tri > 0)]
        sigma = float(np.median(tri)) if tri.size > 0 else 1.0
    else:
        sigma = float(sigma_dir)
    sigma = max(sigma, EPS)
    psi = np.exp(-(dist ** 2) / (sigma ** 2))

    s_dir = np.maximum(0.0, s_role.astype(np.float64)) * psi * kappa
    np.fill_diagonal(s_dir, 0.0)
    return row_topk_sparsify(s_dir.astype(np.float32), top_k, symmetric=False)


def adjacency_to_edge_index(adj: np.ndarray) -> Tensor:
    """无向：去重后双向化（用于对称视图）。"""
    src, dst = np.where(adj > 0)
    mask = src < dst
    src, dst = src[mask], dst[mask]
    if src.size == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    edge_index = np.stack(
        [np.concatenate([src, dst]), np.concatenate([dst, src])],
        axis=0,
    )
    return torch.from_numpy(edge_index).long()


def directed_adjacency_to_edge_index(adj: np.ndarray) -> Tensor:
    """有向：保留 i→j。"""
    src, dst = np.where(adj > 0)
    if src.size == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.from_numpy(np.stack([src, dst], axis=0)).long()


def adjacency_to_edge_index_weight(
    adj: np.ndarray,
    directed: bool = True,
) -> tuple[Tensor, Tensor]:
    """邻接 → (edge_index, edge_weight)。"""
    if directed:
        src, dst = np.where(adj > 0)
        if src.size == 0:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,), dtype=torch.float32)
        w = adj[src, dst].astype(np.float32)
        ei = torch.from_numpy(np.stack([src, dst], axis=0)).long()
        return ei, torch.from_numpy(w).float()

    src, dst = np.where(adj > 0)
    mask = src < dst
    src, dst = src[mask], dst[mask]
    if src.size == 0:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,), dtype=torch.float32)
    w = np.maximum(adj[src, dst], adj[dst, src]).astype(np.float32)
    ei = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])], axis=0)
    ww = np.concatenate([w, w])
    return torch.from_numpy(ei).long(), torch.from_numpy(ww).float()


def count_undirected_edges(adj: np.ndarray) -> int:
    return int((adj > 0).sum() // 2)


def count_directed_edges(adj: np.ndarray) -> int:
    return int((adj > 0).sum())


@dataclass
class MultiViewGraphs:
    adj_phy: np.ndarray
    adj_2nd: np.ndarray
    adj_role: np.ndarray
    adj_dir: np.ndarray
    s_role: np.ndarray
    positions: np.ndarray
    direction_vectors: np.ndarray
    edge_indices: list
    edge_weights: list
    view_names: tuple = VIEW_NAMES

    @property
    def adj_phys(self) -> np.ndarray:
        return self.adj_phy

    @property
    def adj_hop(self) -> np.ndarray:
        return self.adj_2nd

    def edge_counts(self) -> dict:
        phy_mask = self.adj_phy > 0
        dir_mask = self.adj_dir > 0
        return {
            "phy": count_directed_edges(self.adj_phy),
            "2nd": count_directed_edges(self.adj_2nd),
            "role": count_undirected_edges(self.adj_role),
            "dir": count_directed_edges(self.adj_dir),
            "phy_asymmetric_pairs": int((phy_mask != phy_mask.T).sum()),
            "dir_asymmetric_pairs": int((dir_mask != dir_mask.T).sum()),
        }

    def to_device(self, device: torch.device) -> tuple[list, list]:
        eis = [ei.to(device) for ei in self.edge_indices]
        ews = [ew.to(device) for ew in self.edge_weights]
        return eis, ews


def build_multiview_from_adj_phy(
    adj_phy: np.ndarray,
    positions: np.ndarray,
    cfg: Optional[MultiViewGraphConfig] = None,
) -> MultiViewGraphs:
    """
    由给定 A^{phy} 执行 F_mv（式 84–85）：重算 A^{2nd}/A^{role}/A^{dir} 与方向向量。
    坐标 positions 可固定（GPS/MDS）；方向与角色等拓扑量随可见图变化。
    """
    cfg = cfg or MultiViewGraphConfig()
    adj_phy = np.asarray(adj_phy, dtype=np.float32)

    b_2nd = compute_net_second_order(adj_phy, hop_order=cfg.hop_order)
    adj_2nd = build_2nd_adjacency(
        adj_phy,
        hop_order=cfg.hop_order,
        top_k=cfg.view_hop_k if cfg.view_hop_k and cfg.view_hop_k > 0 else None,
        eps=cfg.eps_2nd,
        b_2nd=b_2nd,
    )
    role_feats = build_role_features(adj_phy, b_2nd=b_2nd)
    s_role = role_similarity_matrix(role_feats)
    adj_role = build_role_adjacency(s_role, top_k=cfg.view_role_k)

    direction_vectors = estimate_direction_vectors(positions, adj_phy)
    adj_dir = build_dir_adjacency(
        positions,
        direction_vectors,
        s_role,
        top_k=cfg.view_dir_k,
        sigma_dir=cfg.sigma_dir,
    )

    ei_phy, ew_phy = adjacency_to_edge_index_weight(adj_phy, directed=True)
    ei_2nd, ew_2nd = adjacency_to_edge_index_weight(adj_2nd, directed=True)
    ei_role, ew_role = adjacency_to_edge_index_weight(adj_role, directed=False)
    ei_dir, ew_dir = adjacency_to_edge_index_weight(adj_dir, directed=True)

    return MultiViewGraphs(
        adj_phy=adj_phy,
        adj_2nd=adj_2nd,
        adj_role=adj_role,
        adj_dir=adj_dir,
        s_role=s_role,
        positions=np.asarray(positions, dtype=np.float32),
        direction_vectors=direction_vectors,
        edge_indices=[ei_phy, ei_2nd, ei_role, ei_dir],
        edge_weights=[ew_phy, ew_2nd, ew_role, ew_dir],
    )


def build_multiview_graphs(
    adj_filename: str,
    num_of_vertices: int,
    id_filename: Optional[str] = None,
    cfg: Optional[MultiViewGraphConfig] = None,
    adj_phy: Optional[np.ndarray] = None,
    positions: Optional[np.ndarray] = None,
) -> MultiViewGraphs:
    """构造四视图邻接与 edge_index：A^{phy}, A^{2nd}, A^{role}, A^{dir}。"""
    cfg = cfg or MultiViewGraphConfig()
    if adj_phy is None:
        adj_phy = build_phy_adjacency(
            adj_filename,
            num_of_vertices,
            id_filename=id_filename,
            distance_weight=cfg.phy_distance_weight,
            sigma_d=cfg.sigma_d,
        )
    if positions is None:
        positions = estimate_node_positions(
            adj_filename,
            num_of_vertices,
            id_filename=id_filename,
            coords_filename=cfg.coords_filename,
            seed=cfg.seed,
        )
    return build_multiview_from_adj_phy(adj_phy, positions, cfg=cfg)


def select_views_for_heads(graphs: MultiViewGraphs, num_heads: int) -> MultiViewGraphs:
    """num_heads=1 → 仅 A^{phy}；num_heads>=4 → 全部四视图。"""
    n = max(1, min(int(num_heads), 4))
    names = VIEW_NAMES[:n]
    adjs = [graphs.adj_phy, graphs.adj_2nd, graphs.adj_role, graphs.adj_dir][:n]
    eis = graphs.edge_indices[:n]
    ews = graphs.edge_weights[:n]
    return MultiViewGraphs(
        adj_phy=adjs[0],
        adj_2nd=adjs[1] if n > 1 else graphs.adj_2nd,
        adj_role=adjs[2] if n > 2 else graphs.adj_role,
        adj_dir=adjs[3] if n > 3 else graphs.adj_dir,
        s_role=graphs.s_role,
        positions=graphs.positions,
        direction_vectors=graphs.direction_vectors,
        edge_indices=eis,
        edge_weights=ews,
        view_names=names,
    )
