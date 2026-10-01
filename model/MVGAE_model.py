"""
多视图变分图结构学习（MVGSL）：

  式(90)–(96)：每视图独立 H_0→L_g 层 (Ā)^T 图编码 → {H^m}
  表示层融合 → q_φ(Z|·) → (Z^s, A^{latent}, U)
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.nn.models import InnerProductDecoder
from torch_geometric.utils import negative_sampling as pyg_negative_sampling

MAX_LOGSTD = 10
EPS = 1e-15
# float32 下 sigmoid/BCE 需更大下界，否则 1-EPS 仍可能为 1.0
PROB_EPS = 1e-6

EdgeInput = Union[Tensor, Sequence[Tensor]]
WeightInput = Optional[Union[Tensor, Sequence[Optional[Tensor]]]]


def _as_edge_list(edge_index: EdgeInput) -> list[Tensor]:
    if isinstance(edge_index, (list, tuple)):
        return list(edge_index)
    return [edge_index]


def _as_weight_list(
    edge_weight: WeightInput,
    num_views: int,
) -> list[Optional[Tensor]]:
    if edge_weight is None:
        return [None] * num_views
    if isinstance(edge_weight, (list, tuple)):
        ws = list(edge_weight)
        if len(ws) != num_views:
            raise ValueError(
                "edge_weights 数量 %d 与 num_views=%d 不一致" % (len(ws), num_views)
            )
        return ws
    return [edge_weight] + [None] * (num_views - 1)


def row_normalize_edges(
    edge_index: Tensor,
    edge_weight: Optional[Tensor],
    num_nodes: int,
    add_self_loops: bool = True,
    renormalize: bool = True,
) -> tuple[Tensor, Tensor]:
    """
    行归一化边权。
    - phy（式35–37）：add_self_loops=True → Â=(D^{out})^{-1}(A+I)
    - 2nd（式42–43）：构图已完成 (D+εI)^{-1}B，此处 add_self_loops=False, renormalize=False
    """
    device = edge_index.device
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=device, dtype=torch.float32)
    else:
        edge_weight = edge_weight.to(device=device, dtype=torch.float32)

    if add_self_loops:
        loop = torch.arange(num_nodes, device=device)
        loop_index = torch.stack([loop, loop], dim=0)
        edge_index = torch.cat([edge_index, loop_index], dim=1)
        edge_weight = torch.cat(
            [edge_weight, torch.ones(num_nodes, device=device, dtype=edge_weight.dtype)],
            dim=0,
        )

    if renormalize:
        row = edge_index[0]
        deg = torch.zeros(num_nodes, device=device, dtype=edge_weight.dtype)
        deg.scatter_add_(0, row, edge_weight)
        deg = deg.clamp_min(EPS)
        edge_weight = edge_weight / deg[row]
    return edge_index, edge_weight


class RowNormConv(MessagePassing):
    """
    式(93)(94) 有向图卷积：H' = (Ā)^T (H W) + b。

    边 (i→j) 权重 Ā_ij：消息从源 i 传到目标 j，
    h'_j ← Σ_i Ā_ij (h_i W)，即仅有 i→j 关系的节点参与 j 的聚合。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        add_self_loops: bool = True,
        renormalize: bool = True,
    ):
        # source_to_target：在目标节点 j 上聚合来自源节点 i 的消息
        super().__init__(aggr="add", flow="source_to_target")
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)
        self.add_self_loops = bool(add_self_loops)
        self.renormalize = bool(renormalize)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        # 先 W_l，再 (Ā)^T 聚合，对应式(94)：Σ_i Ā_ij h_i W_l + b_l
        x = self.lin(x)
        edge_index, edge_weight = row_normalize_edges(
            edge_index,
            edge_weight,
            x.size(0),
            add_self_loops=self.add_self_loops,
            renormalize=self.renormalize,
        )
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        # flow=source_to_target 时 x_j 为源节点特征
        return edge_weight.view(-1, 1) * x_j


class SharedGCNBackbone(nn.Module):
    """旧接口：对称归一化 GCN（邻接学习等兼容路径）。"""

    def __init__(self, in_channels: int, hidden_channels: int, num_gcn_layers: int = 2):
        super().__init__()
        if num_gcn_layers < 1:
            raise ValueError("num_gcn_layers 至少为 1")

        convs: list[GCNConv] = []
        convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_gcn_layers - 1):
            convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs = nn.ModuleList(convs)
        self.hidden_channels = hidden_channels

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = F.relu(h)
        return h


class RowNormGCNBackbone(nn.Module):
    """
    单视图结构编码器（式 92–95）：
      H_0 = φ(X_vis W_in + b_in)
      H_{l+1} = φ[(Ā_vis)^T H_l W_l + b_l]，l = 0..L_g-1
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_gcn_layers: int = 2,
        add_self_loops: bool = True,
        renormalize: bool = True,
    ):
        super().__init__()
        if num_gcn_layers < 1:
            raise ValueError("num_gcn_layers 至少为 1（对应 L_g）")
        # 式(92)：每视图独立输入映射
        self.in_proj = nn.Linear(in_channels, hidden_channels)
        # 式(93)：L_g 层图编码（hidden → hidden）
        self.convs = nn.ModuleList(
            [
                RowNormConv(
                    hidden_channels,
                    hidden_channels,
                    add_self_loops=add_self_loops,
                    renormalize=renormalize,
                )
                for _ in range(num_gcn_layers)
            ]
        )
        self.hidden_channels = hidden_channels
        self.num_gcn_layers = int(num_gcn_layers)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        h = F.relu(self.in_proj(x))  # H_0^{(m)}
        for conv in self.convs:
            h = F.relu(conv(h, edge_index, edge_weight))  # H_{l+1}^{(m)}
        return h


class VariationalHead(nn.Module):
    """单头变分投影：h → (μ, logstd)。"""

    def __init__(self, hidden_channels: int, out_channels: int):
        super().__init__()
        self.mu_proj = nn.Linear(hidden_channels, out_channels)
        self.logstd_proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        return self.mu_proj(h), self.logstd_proj(h)


class DeterministicHead(nn.Module):
    """单头确定性投影（经典 GAE）：h → μ；logstd 返回全零占位。"""

    def __init__(self, hidden_channels: int, out_channels: int):
        super().__init__()
        self.mu_proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        mu = self.mu_proj(h)
        return mu, torch.zeros_like(mu)


class ViewFusion(nn.Module):
    """
    节点级视图注意力融合（式 97–102）：
      e_i^{(m)} = q_v^T tanh(W_v h_i^{(m)} + b_v)
      α_i = softmax_m(e_i)
      h_i^{mv} = Σ_m α_i^{(m)} h_i^{(m)}
    """

    def __init__(
        self,
        num_views: int,
        hidden_dim: int,
        out_dim: Optional[int] = None,
        attn_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_views = num_views
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim if out_dim is not None else hidden_dim
        d_a = int(attn_dim) if attn_dim is not None else hidden_dim
        # 式(97)：W_v, b_v, q_v（各视图共享）
        self.W_v = nn.Linear(hidden_dim, d_a, bias=True)
        self.q_v = nn.Parameter(torch.empty(d_a))
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.zeros_(self.W_v.bias)
        nn.init.xavier_uniform_(self.q_v.unsqueeze(0))
        if self.out_dim != hidden_dim:
            self.proj = nn.Linear(hidden_dim, self.out_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, h_views: Tensor) -> Tensor:
        """
        h_views : [N, V, H]
        returns H^{mv} : [N, out_dim]
        """
        # e_i^{(m)} = q_v^T tanh(W_v h_i^{(m)} + b_v)  → [N, V]
        scores = torch.tanh(self.W_v(h_views))
        e = torch.matmul(scores, self.q_v)
        alpha = F.softmax(e, dim=1)
        fused = (h_views * alpha.unsqueeze(-1)).sum(dim=1)
        return self.proj(fused)

    def attention_weights(self, h_views: Tensor) -> Tensor:
        """返回节点级视图权重 α : [N, V]，便于分析式(101)。"""
        scores = torch.tanh(self.W_v(h_views))
        e = torch.matmul(scores, self.q_v)
        return F.softmax(e, dim=1)


# 兼容旧名
HeadFusion = ViewFusion


def negative_sampling_edges(
    pos_edge_index: Tensor,
    num_nodes: int,
    num_neg: int,
    device: Optional[torch.device] = None,
) -> Tensor:
    """在避开正样本的前提下采样有向负边。"""
    if device is None:
        device = pos_edge_index.device
    neg = pyg_negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=max(1, int(num_neg)),
    )
    return neg.to(device)


class AsymmetricBilinearDecoder(nn.Module):
    """
    源—目标非对称双线性解码（式 114–117 / 128）：
      z_i^{src}=W_src z_i,  z_j^{dst}=W_dst z_j
      s_ij = (z_i^{src})^T z_j^{dst} / d_z
      A_ij^{latent} = sigmoid(s_ij)
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.W_src = nn.Linear(latent_dim, latent_dim, bias=False)
        self.W_dst = nn.Linear(latent_dim, latent_dim, bias=False)
        nn.init.xavier_uniform_(self.W_src.weight)
        nn.init.xavier_uniform_(self.W_dst.weight)

    def score(self, z: Tensor, edge_index: Tensor) -> Tensor:
        zi = self.W_src(z)[edge_index[0]]
        zj = self.W_dst(z)[edge_index[1]]
        return (zi * zj).sum(dim=-1) / float(self.latent_dim)

    def forward(self, z: Tensor, edge_index: Tensor, sigmoid: bool = True) -> Tensor:
        s = self.score(z, edge_index).clamp(-20.0, 20.0)
        return torch.sigmoid(s) if sigmoid else s

    def dense(self, z: Tensor) -> Tensor:
        """稠密 A^{latent} ∈ R^{N×N}，对角清零。"""
        zs = self.W_src(z)
        zd = self.W_dst(z)
        s = ((zs @ zd.T) / float(self.latent_dim)).clamp(-20.0, 20.0)
        adj = torch.sigmoid(s)
        adj.fill_diagonal_(0.0)
        return adj


class MultiViewBilinearDecoder(nn.Module):
    """
    多视图辅助关系解码（式 122–124）：
      s_ij^m = z_i^T R_m z_j,  A_ij^m = sigmoid(s_ij^m)
      role 视图：R_role ← (B + B^T)/2
    """

    def __init__(
        self,
        latent_dim: int,
        num_views: int,
        view_names: Optional[Sequence[str]] = None,
        symmetric_views: Sequence[str] = ("role",),
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.num_views = int(num_views)
        default_views = ("phy", "2nd", "role", "dir")
        names = list(view_names) if view_names is not None else list(default_views[:num_views])
        if len(names) != num_views:
            names = [f"view{i}" for i in range(num_views)]
        self.view_names = names
        self.symmetric = {i for i, n in enumerate(names) if n in set(symmetric_views)}
        self.R = nn.ParameterList()
        for _ in range(num_views):
            p = nn.Parameter(torch.empty(latent_dim, latent_dim))
            nn.init.xavier_uniform_(p, gain=0.1)
            self.R.append(p)

    def relation_matrix(self, view_idx: int) -> Tensor:
        r = self.R[view_idx]
        if view_idx in self.symmetric:
            return 0.5 * (r + r.T)
        return r

    def forward(
        self,
        z: Tensor,
        edge_index: Tensor,
        view_idx: int,
        sigmoid: bool = True,
    ) -> Tensor:
        r = self.relation_matrix(view_idx)
        zi = z[edge_index[0]]
        zj = z[edge_index[1]]
        s = ((zi @ r) * zj).sum(dim=-1).clamp(-20.0, 20.0)
        return torch.sigmoid(s) if sigmoid else s


class MultiViewGCNEncoder(nn.Module):
    """
    多视图结构编码（式 90–96）+ MVGSL 融合后验（式 76）：
      每视图独立：H_0^{(m)}=φ(XW_in^{(m)}) → L_g 层 (Ā^{(m)})^T 图编码 → H^{(m)}
      ViewFusion：{H^m} → H^{mv} → q_φ(Z|·)
    """

    _VIEW_NORM = {
        "phy": (True, True),
        "2nd": (False, False),
        "role": (True, True),
        "dir": (True, True),
    }

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_heads: int,
        num_gcn_layers: int = 2,
        fusion_out_dim: Optional[int] = None,
        variational: bool = True,
        view_names: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads/num_views 至少为 1")

        self.num_heads = num_heads
        self.num_views = num_heads
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.variational = bool(variational)
        default_views = ("phy", "2nd", "role", "dir")
        names = list(view_names) if view_names is not None else list(default_views[:num_heads])
        if len(names) != num_heads:
            names = [f"view{i}" for i in range(num_heads)]
        self.view_names = names

        backbones = []
        for name in names:
            add_loop, renorm = self._VIEW_NORM.get(name, (True, True))
            backbones.append(
                RowNormGCNBackbone(
                    in_channels,
                    hidden_channels,
                    num_gcn_layers,
                    add_self_loops=add_loop,
                    renormalize=renorm,
                )
            )
        self.backbones = nn.ModuleList(backbones)
        # 先融 H^v → H^{mv}（维度保持 hidden），再变分投影到 latent d_z
        self.view_fusion = ViewFusion(num_heads, hidden_channels, hidden_channels)
        head_cls = VariationalHead if self.variational else DeterministicHead
        self.posterior = head_cls(hidden_channels, out_channels)
        # 式(111)(112)：Z^s = μ ∈ R^{N×d_z}，不再做额外维数投影
        if fusion_out_dim is not None and int(fusion_out_dim) != int(out_channels):
            import warnings

            warnings.warn(
                "fusion_out_dim=%s 与 latent(d_z)=%s 不一致；"
                "已按式(111)强制 Z^s=μ，输出维数为 latent。"
                % (fusion_out_dim, out_channels),
                UserWarning,
                stacklevel=2,
            )
        self.z_proj = nn.Identity()
        self._fusion_out_dim = int(out_channels)
        # 兼容旧属性名
        self.fusion = self.view_fusion

    @property
    def fusion_dim(self) -> int:
        return self._fusion_out_dim

    def encode_views(
        self,
        x: Tensor,
        edge_indices: EdgeInput,
        edge_weights: WeightInput = None,
    ) -> Tensor:
        """返回 H_views : [N, V, H]。"""
        eis = _as_edge_list(edge_indices)
        ews = _as_weight_list(edge_weights, self.num_views)
        if len(eis) != self.num_views:
            raise ValueError(
                "edge_indices 数量 %d 与 num_views=%d 不一致" % (len(eis), self.num_views)
            )
        hs = [backbone(x, ei, ew) for backbone, ei, ew in zip(self.backbones, eis, ews)]
        return torch.stack(hs, dim=1)

    def forward(
        self,
        x: Tensor,
        edge_indices: EdgeInput,
        edge_weights: WeightInput = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Returns
        -------
        mu : [N, latent]
        logstd : [N, latent]
        z_s : [N, latent]  稳定结构表示 Z^s=μ（式 111）
        h_views : [N, V, H]
        h_mv : [N, H]
        """
        h_views = self.encode_views(x, edge_indices, edge_weights)
        h_mv = self.view_fusion(h_views)
        mu, logstd = self.posterior(h_mv)
        z_s = mu  # 式(111)：z_i^s = μ_i
        return mu, logstd, z_s, h_views, h_mv


class MVGAEEncoderHead(nn.Module):
    """兼容旧接口：独立 GCN 头（保留供邻接学习脚本使用）。"""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_gcn_layers: int = 2,
    ):
        super().__init__()
        self.backbone = SharedGCNBackbone(in_channels, hidden_channels, num_gcn_layers)
        self.head = VariationalHead(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        h = self.backbone(x, edge_index)
        return self.head(h)


class MultiHeadGCNEncoder(nn.Module):
    """兼容旧接口：num_heads 路独立 GCN 头。"""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_heads: int,
        num_gcn_layers: int = 2,
    ):
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads 至少为 1")
        self.num_heads = num_heads
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.variational = True
        self.heads = nn.ModuleList(
            [
                MVGAEEncoderHead(
                    in_channels,
                    hidden_channels,
                    out_channels,
                    num_gcn_layers=num_gcn_layers,
                )
                for _ in range(num_heads)
            ]
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        mus: list[Tensor] = []
        logstds: list[Tensor] = []
        for head in self.heads:
            mu, logstd = head(x, edge_index)
            mus.append(mu)
            logstds.append(logstd)
        return torch.stack(mus, dim=1), torch.stack(logstds, dim=1)


class MVGAE(nn.Module):
    """
    多视图 VGAE / 单视图 GAE。
    - shared_backbone=True：多视图编码 + 非对称潜在解码 + 多视图辅助重构；
    - shared_backbone=False：独立多头（邻接学习兼容模式）；
    - variational=False：确定性编码（无 KL / 无重参数化采样）。
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: Optional[nn.Module] = None,
        shared_backbone: bool = False,
        variational: Optional[bool] = None,
        view_decoder: Optional[MultiViewBilinearDecoder] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.shared_backbone = shared_backbone
        self.num_heads = encoder.num_heads
        if variational is None:
            variational = bool(getattr(encoder, "variational", True))
        self.variational = bool(variational)

        latent_dim = (
            int(getattr(encoder, "out_channels", encoder.fusion_dim))
            if shared_backbone
            else int(encoder.out_channels)
        )
        if decoder is None:
            decoder = (
                AsymmetricBilinearDecoder(latent_dim)
                if shared_backbone
                else InnerProductDecoder()
            )
        self.decoder = decoder

        if view_decoder is None and shared_backbone:
            view_decoder = MultiViewBilinearDecoder(
                latent_dim,
                num_views=int(encoder.num_heads),
                view_names=getattr(encoder, "view_names", None),
            )
        self.view_decoder = view_decoder

        self.__mu__: Tensor | None = None
        self.__logstd__: Tensor | None = None
        self.__z_init__: Tensor | None = None
        self.__h_views__: Tensor | None = None
        self.__h_mv__: Tensor | None = None
        self.__a_latent__: Tensor | None = None
        self.__u__: Tensor | None = None
        self.__c_v__: Tensor | None = None

    @property
    def latent_dim(self) -> int:
        if self.shared_backbone:
            return self.encoder.fusion_dim
        return self.encoder.out_channels * self.num_heads

    def reset_parameters(self) -> None:
        for m in self.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def reparametrize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        if not self.variational:
            return mu
        logstd = logstd.clamp(max=MAX_LOGSTD)
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        return mu

    def latent_adjacency(self, z: Tensor) -> Tensor:
        """式(117)/(128)：非对称 A^{latent}；无非对称解码器时回退对称内积。"""
        if isinstance(self.decoder, AsymmetricBilinearDecoder):
            return self.decoder.dense(z)
        adj = torch.sigmoid(z @ z.T)
        adj.fill_diagonal_(0.0)
        return adj

    @staticmethod
    def node_uncertainty(logstd: Tensor) -> Tensor:
        """式(131)：节点级结构不确定性 u_i = mean_r(σ_{i,r})。"""
        return torch.exp(logstd.clamp(max=MAX_LOGSTD)).mean(dim=-1)

    @staticmethod
    def structure_confidence(u: Tensor) -> Tensor:
        """式(132)–(133)：节点结构可信度 c_i = exp(-u_i) ∈ (0,1]。"""
        return torch.exp(-u.clamp_min(0.0))

    def encode(
        self,
        x: Tensor,
        edge_index: EdgeInput,
        edge_weight: WeightInput = None,
    ) -> Tensor:
        if self.shared_backbone:
            mu, logstd, z_s, h_views, h_mv = self.encoder(x, edge_index, edge_weight)
            if self.variational:
                logstd = logstd.clamp(max=MAX_LOGSTD)
            self.__mu__ = mu
            self.__logstd__ = logstd
            self.__z_init__ = z_s
            self.__h_views__ = h_views
            self.__h_mv__ = h_mv
            z = self.reparametrize(mu, logstd)
            # 预训练阶段可用采样 z；导出阶段改用 μ（encode_z_init）
            self.__a_latent__ = self.latent_adjacency(z)
            self.__u__ = self.node_uncertainty(logstd)
            self.__c_v__ = self.structure_confidence(self.__u__)
            return z
        ei = _as_edge_list(edge_index)[0]
        self.__mu__, self.__logstd__ = self.encoder(x, ei)
        if self.variational:
            self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        return self.reparametrize(self.__mu__, self.__logstd__)

    @property
    def z_init(self) -> Tensor:
        if self.__z_init__ is None:
            raise RuntimeError("请先调用 encode()")
        return self.__z_init__

    @property
    def z_s(self) -> Tensor:
        return self.z_init

    @property
    def a_latent(self) -> Tensor:
        if self.__a_latent__ is None:
            raise RuntimeError("请先调用 encode() / encode_structure()")
        return self.__a_latent__

    @property
    def uncertainty(self) -> Tensor:
        if self.__u__ is None:
            raise RuntimeError("请先调用 encode() / encode_structure()")
        return self.__u__

    @property
    def confidence(self) -> Tensor:
        """结构可信度 C_v（式 134）。"""
        if self.__c_v__ is None:
            raise RuntimeError("请先调用 encode() / encode_structure()")
        return self.__c_v__

    def encode_structure(
        self,
        x: Tensor,
        edge_index: EdgeInput,
        edge_weight: WeightInput = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """返回 (Z^s, A^{latent}, U, C_v)，式(135)。"""
        self.encode_z_init(x, edge_index, edge_weight)
        assert self.__z_init__ is not None and self.__a_latent__ is not None
        assert self.__u__ is not None and self.__c_v__ is not None
        return self.__z_init__, self.__a_latent__, self.__u__, self.__c_v__

    def compute_z_init(
        self,
        x: Tensor,
        edge_index: EdgeInput,
        edge_weight: WeightInput = None,
    ) -> Tensor:
        """推理阶段获取 Z^s（使用 μ；无梯度）。"""
        self.eval()
        with torch.no_grad():
            return self.encode_z_init(x, edge_index, edge_weight)

    def encode_z_init(
        self,
        x: Tensor,
        edge_index: EdgeInput,
        edge_weight: WeightInput = None,
    ) -> Tensor:
        """可微稳定结构表示 Z^s。"""
        if self.shared_backbone:
            mu, logstd, z_s, h_views, h_mv = self.encoder(x, edge_index, edge_weight)
            self.__mu__ = mu
            self.__logstd__ = logstd.clamp(max=MAX_LOGSTD) if self.variational else logstd
            self.__z_init__ = z_s
            self.__h_views__ = h_views
            self.__h_mv__ = h_mv
            self.__a_latent__ = self.latent_adjacency(z_s)
            self.__u__ = self.node_uncertainty(self.__logstd__)
            self.__c_v__ = self.structure_confidence(self.__u__)
            return z_s
        ei = _as_edge_list(edge_index)[0]
        mu, logstd = self.encoder(x, ei)
        self.__mu__ = mu
        self.__logstd__ = logstd
        self.__z_init__ = mu.reshape(mu.size(0), -1)
        return self.__z_init__

    def kl_loss(
        self,
        mu: Optional[Tensor] = None,
        logstd: Optional[Tensor] = None,
    ) -> Tensor:
        """
        式(139)(140)：L_KL = (1/N) Σ_i D_KL(q(z_i)||N(0,I))。
        logstd := log σ，故 log(σ²)=2 logstd，σ²=exp(logstd)²。
        """
        if not self.variational:
            mu = self.__mu__ if mu is None else mu
            assert mu is not None
            return mu.new_zeros(())
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(max=MAX_LOGSTD)
        assert mu is not None and logstd is not None
        kl = -0.5 * torch.sum(1 + 2 * logstd - mu.pow(2) - logstd.exp().pow(2), dim=-1)
        return kl.mean()

    def diversity_loss(self, h_views: Optional[Tensor] = None) -> Tensor:
        """视图表示多样性：惩罚不同视图 H^v 的余弦相似度。"""
        h = self.__h_views__ if h_views is None else h_views
        if h is None:
            mu = self.__mu__
            if mu is None:
                return torch.zeros(())
            if mu.dim() != 3:
                return mu.new_zeros(())
            h = mu
        num_views = h.size(1)
        if num_views < 2:
            return h.new_zeros(())
        loss = h.new_zeros(())
        count = 0
        for i in range(num_views):
            zi = F.normalize(h[:, i], dim=-1)
            for j in range(i + 1, num_views):
                zj = F.normalize(h[:, j], dim=-1)
                loss = loss + (zi * zj).sum(dim=-1).pow(2).mean()
                count += 1
        return loss / max(count, 1)

    def _head_recon_loss(
        self,
        z: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
        lambda_neg: float = 1.0,
    ) -> Tensor:
        """式(121) 形式的正负边 BCE（解码器可为非对称）。"""
        if pos_edge_index.numel() == 0:
            return z.new_zeros(())
        pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True).clamp(PROB_EPS, 1.0 - PROB_EPS)).mean()
        if neg_edge_index.numel() == 0:
            return pos_loss
        neg_loss = -torch.log(
            (1 - self.decoder(z, neg_edge_index, sigmoid=True)).clamp(PROB_EPS, 1.0 - PROB_EPS)
        ).mean()
        return pos_loss + float(lambda_neg) * neg_loss

    def mask_recon_loss(
        self,
        z: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
        lambda_neg: float = 1.0,
    ) -> Tensor:
        """式(119)–(121)：被遮蔽物理关系重构 L_mask。"""
        return self._head_recon_loss(z, pos_edge_index, neg_edge_index, lambda_neg=lambda_neg)

    def multiview_recon_loss(
        self,
        z: Tensor,
        edge_indices: Sequence[Tensor],
        edge_weights: Sequence[Optional[Tensor]],
        num_nodes: int,
        omega: Optional[Sequence[float]] = None,
        subsample: int = 2048,
        neg_ratio: int = 1,
    ) -> Tensor:
        """
        式(122)–(126)：多视图辅助重构 L_mv。
        软标签取各视图归一化边权；Ω_m = 正关系 ∪ 采样零关系。
        """
        if self.view_decoder is None:
            return z.new_zeros(())
        num_views = self.view_decoder.num_views
        if omega is None:
            w = [1.0 / num_views] * num_views
        else:
            w = [float(x) for x in omega]
            s = sum(w) + EPS
            w = [x / s for x in w]

        total = z.new_zeros(())
        for m in range(min(num_views, len(edge_indices))):
            ei = edge_indices[m]
            ew = edge_weights[m] if m < len(edge_weights) else None
            if ei is None or ei.numel() == 0:
                continue
            n_pos = int(ei.size(1))
            if n_pos > subsample:
                idx = torch.randperm(n_pos, device=z.device)[:subsample]
                ei_pos = ei[:, idx]
                y_pos = (
                    ew[idx].clamp(0.0, 1.0)
                    if ew is not None
                    else z.new_ones(ei_pos.size(1))
                )
            else:
                ei_pos = ei
                y_pos = (
                    ew.clamp(0.0, 1.0)
                    if ew is not None
                    else z.new_ones(ei_pos.size(1))
                )

            n_neg = max(1, int(ei_pos.size(1) * neg_ratio))
            ei_neg = negative_sampling_edges(ei, num_nodes, n_neg, device=z.device)
            y_neg = z.new_zeros(ei_neg.size(1))

            ei_all = torch.cat([ei_pos, ei_neg], dim=1)
            y_all = torch.cat([y_pos, y_neg], dim=0)
            pred = self.view_decoder(z, ei_all, m, sigmoid=True).clamp(PROB_EPS, 1.0 - PROB_EPS)
            # 式(126)：软标签 BCE
            y_all = y_all.clamp(0.0, 1.0)
            bce = -(y_all * pred.log() + (1.0 - y_all) * (1.0 - pred).log()).mean()
            if not torch.isfinite(bce):
                continue
            total = total + w[m] * bce
        return total

    def structure_recon_loss(
        self,
        z: Tensor,
        mask_pos: Tensor,
        mask_neg: Tensor,
        edge_indices: Sequence[Tensor],
        edge_weights: Sequence[Optional[Tensor]],
        num_nodes: int,
        lambda_neg: float = 1.0,
        lambda_mv: float = 1.0,
        omega: Optional[Sequence[float]] = None,
        mv_subsample: int = 2048,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """式(127)：L_rec = L_mask + λ_mv L_mv。返回 (L_rec, L_mask, L_mv)。"""
        loss_mask = self.mask_recon_loss(z, mask_pos, mask_neg, lambda_neg=lambda_neg)
        loss_mv = self.multiview_recon_loss(
            z,
            edge_indices,
            edge_weights,
            num_nodes=num_nodes,
            omega=omega,
            subsample=mv_subsample,
        )
        loss_rec = loss_mask + float(lambda_mv) * loss_mv
        return loss_rec, loss_mask, loss_mv

    def recon_loss(
        self,
        z: Tensor,
        pos_edge_index: EdgeInput,
        neg_edge_index: Optional[EdgeInput] = None,
        lambda_neg: float = 1.0,
    ) -> Tensor:
        if neg_edge_index is None:
            raise ValueError("MVGAE.recon_loss 需要显式传入 neg_edge_index")

        pos_list = _as_edge_list(pos_edge_index)
        neg_list = _as_edge_list(neg_edge_index)

        if z.dim() == 2:
            losses = [
                self._head_recon_loss(z, pos_list[i], neg_list[i], lambda_neg=lambda_neg)
                for i in range(len(pos_list))
            ]
            return torch.stack(losses).mean()

        if len(pos_list) == z.size(1) and len(neg_list) == z.size(1):
            losses = [
                self._head_recon_loss(z[:, h], pos_list[h], neg_list[h], lambda_neg=lambda_neg)
                for h in range(z.size(1))
            ]
            return torch.stack(losses).mean()

        losses = [
            self._head_recon_loss(z[:, h], pos_list[0], neg_list[0], lambda_neg=lambda_neg)
            for h in range(z.size(1))
        ]
        return torch.stack(losses).mean()

    def _fused_edge_pred(self, z: Tensor, edge_index: Tensor) -> Tensor:
        preds = torch.stack(
            [self.decoder(z[:, h], edge_index, sigmoid=True) for h in range(z.size(1))],
            dim=0,
        )
        return preds.mean(dim=0)

    @torch.no_grad()
    def test(
        self,
        z: Tensor,
        pos_edge_index: Tensor,
        neg_edge_index: Tensor,
    ) -> tuple[float, float]:
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score
        except ImportError as e:
            raise ImportError("评估 AUC/AP 需要 scikit-learn") from e

        if z.dim() == 2:
            pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
            neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        else:
            pos_pred = self._fused_edge_pred(z, pos_edge_index)
            neg_pred = self._fused_edge_pred(z, neg_edge_index)

        pos_pred = torch.nan_to_num(pos_pred, nan=0.5).clamp(0.0, 1.0)
        neg_pred = torch.nan_to_num(neg_pred, nan=0.5).clamp(0.0, 1.0)
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0).cpu().numpy()
        pred = torch.cat([pos_pred, neg_pred], dim=0).cpu().numpy()
        return float(roc_auc_score(y, pred)), float(average_precision_score(y, pred))

    def encode_fused(
        self,
        x: Tensor,
        edge_index: EdgeInput,
        edge_weight: WeightInput = None,
    ) -> Tensor:
        """返回融合嵌入 Z_init。"""
        if self.shared_backbone:
            return self.compute_z_init(x, edge_index, edge_weight)
        z = self.encode(x, edge_index, edge_weight)
        if z.dim() == 3:
            return z.reshape(z.size(0), -1)
        return z

    @torch.no_grad()
    def predict_adjacency(
        self,
        x: Tensor,
        edge_index: EdgeInput,
        symmetrize: bool = False,
        edge_weight: WeightInput = None,
    ) -> Tensor:
        self.eval()
        z = self.encode_z_init(x, edge_index, edge_weight)
        adj = self.latent_adjacency(z)
        if symmetrize:
            adj = (adj + adj.T) * 0.5
            adj.fill_diagonal_(0.0)
        return adj


def build_shared_mvgae(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_heads: int = 4,
    num_gcn_layers: int = 2,
    fusion_out_dim: Optional[int] = None,
    variational: bool = True,
) -> MVGAE:
    """构建多视图 MVGAE / 单视图 GAE（预训练阶段）。num_heads ≡ num_views。"""
    encoder = MultiViewGCNEncoder(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_heads=num_heads,
        num_gcn_layers=num_gcn_layers,
        fusion_out_dim=fusion_out_dim,
        variational=variational,
    )
    decoder = AsymmetricBilinearDecoder(out_channels)
    view_decoder = MultiViewBilinearDecoder(
        out_channels,
        num_views=num_heads,
        view_names=getattr(encoder, "view_names", None),
    )
    return MVGAE(
        encoder,
        decoder=decoder,
        shared_backbone=True,
        variational=variational,
        view_decoder=view_decoder,
    )


def build_pyg_mvgae(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_heads: int = 4,
    num_gcn_layers: int = 2,
) -> MVGAE:
    """构建独立多头 MVGAE（邻接学习兼容）。"""
    encoder = MultiHeadGCNEncoder(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_heads=num_heads,
        num_gcn_layers=num_gcn_layers,
    )
    return MVGAE(encoder, shared_backbone=False, variational=True)
