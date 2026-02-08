import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GraphNorm
from torch_geometric.utils import softmax, dropout_edge
import math

# ==========================================
# 0. 基础组件: DropPath (随机深度)
# ==========================================
class DropPath(nn.Module):
    """Stochastic Depth: 在训练时随机丢弃残差路径，增强泛化能力"""
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_() 
        return x.div(keep_prob) * random_tensor

# ==========================================
# 1. 新增组件: Entropy-Aware Gating (并行融合门控)
# ==========================================
class EntropyGatingUnit(nn.Module):
    """
    自适应门控单元：
    根据流的全局特征自动调整 局部特征(Inception) 与 全局特征(Attention) 的权重。
    包含 Residual Connection 以保证梯度流的稳定性。
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # 轻量级 MLP 用于计算门控系数 alpha
        # 隐含层维度减半以节省参数
        self.gate_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x_local, x_global, x_base):
        """
        Args:
            x_local:  [Batch, Time, Hidden] - 来自 Inception 流
            x_global: [Batch, Time, Hidden] - 来自 Linear Attention 流
            x_base:   [Batch, Time, Hidden] - 原始输入 (用于残差连接)
        Returns:
            out: 融合后的特征
            alpha: 门控权重 (用于可视化分析)
        """
        # 1. 计算当前流的全局特征摘要 (这里使用平均池化)
        # 这模拟了从整个流中提取 "熵" 或 "统计分布" 的过程
        flow_embedding = x_base.mean(dim=1) # [Batch, Hidden]
        
        # 2. 计算门控系数 alpha [Batch, 1, 1]
        # alpha 接近 1 -> 偏向局部特征 (如 DDoS)
        # alpha 接近 0 -> 偏向全局特征 (如 Tor)
        alpha = self.gate_fc(flow_embedding).unsqueeze(1) 
        
        # 3. 加权融合 + 残差连接 (Robustness)
        # 公式: Y = alpha * Local + (1-alpha) * Global + Base
        out = alpha * x_local + (1 - alpha) * x_global + x_base
        
        return out, alpha

# ==========================================
# 2. 核心组件: Temporal Inception 1D (Local Stream)
# ==========================================
class TemporalInception1D(nn.Module):
    """
    多尺度时序卷积模块 (Stream A):
    同时捕捉短周期（k=1,3）和长周期（k=5,7）的局部流量特征。
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.kernel_set = [1, 3, 5, 7] 
        cout_per_kernel = out_features // len(self.kernel_set)
        
        self.tconv = nn.ModuleList()
        for kern in self.kernel_set:
            pad = kern // 2
            self.tconv.append(
                nn.Conv1d(in_features, cout_per_kernel, kernel_size=kern, padding=pad)
            )
        
        self.project = nn.Conv1d(in_features, out_features, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [Batch, Hidden, Time]
        outputs = []
        for conv in self.tconv:
            outputs.append(conv(x))
        out = torch.cat(outputs, dim=1) 
        
        # 残差连接 (Inception 内部)
        return self.act(out + self.project(x))

# ==========================================
# 3. 核心组件: Edge-Augmented Attention (Phase 1)
# ==========================================
class EdgeAugmentedAttention(MessagePassing):
    """
    边增强注意力机制：
    将网络流量的边特征（Edge Features）显式融入节点的消息传递中。
    """
    def __init__(self, in_dim, out_dim, edge_dim, heads=4, dropout=0.1, drop_path=0.1):
        super().__init__(node_dim=0, aggr='add')
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.heads = heads
        self.head_dim = out_dim // heads
        self.dropout = dropout

        assert out_dim % heads == 0, "out_dim must be divisible by heads"

        self.WQ = nn.Linear(in_dim, out_dim, bias=False)
        self.WK = nn.Linear(in_dim, out_dim, bias=False)
        self.WV = nn.Linear(in_dim, out_dim, bias=False)
        self.WE = nn.Linear(edge_dim, out_dim, bias=False)

        self.out_proj = nn.Linear(out_dim, out_dim)
        self.norm = GraphNorm(out_dim)
        self.drop_path = DropPath(drop_path)
        self.act = nn.GELU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.WQ.weight)
        nn.init.xavier_uniform_(self.WK.weight)
        nn.init.xavier_uniform_(self.WV.weight)
        nn.init.xavier_uniform_(self.WE.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, edge_index, edge_attr, batch=None):
        residual = x
        q = self.WQ(x).view(-1, self.heads, self.head_dim)
        k = self.WK(x).view(-1, self.heads, self.head_dim)
        v = self.WV(x).view(-1, self.heads, self.head_dim)
        e_emb = self.WE(edge_attr).view(-1, self.heads, self.head_dim)

        out = self.propagate(edge_index, q=q, k=k, v=v, e_emb=e_emb, size=None)
        
        out = out.view(-1, self.out_dim)
        out = self.out_proj(out)
        out = self.norm(out + self.drop_path(residual), batch)
        return self.act(out)

    def message(self, q_i, k_j, v_j, e_emb, index):
        score = (q_i * (k_j + e_emb)).sum(dim=-1) / (self.head_dim ** 0.5)
        alpha = softmax(score, index)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha.unsqueeze(-1) * (v_j + e_emb)

# ==========================================
# 4. 核心组件: Edge Updater (Phase 1)
# ==========================================
class EdgeUpdaterModule(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, dropout=0.1):
        super().__init__()
        input_dim = node_dim * 2 + edge_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.res_proj = nn.Linear(edge_dim, hidden_dim) if edge_dim != hidden_dim else None
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        x_src, x_dst = x[src], x[dst]
        cat_feat = torch.cat([x_src, x_dst, edge_attr], dim=-1)
        
        update = self.mlp(cat_feat)
        
        if self.res_proj is not None:
            edge_attr = self.res_proj(edge_attr)
            
        return self.norm(update + edge_attr)

# ==========================================
# 5. 核心组件: Linear Temporal Attention (Global Stream)
# ==========================================
class LinearTemporalAttention(nn.Module):
    """
    Stream B: 全局时序注意力
    使用 O(T) 复杂度的线性 Attention 捕捉长程依赖。
    """
    def __init__(self, feature_dim, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.head_dim = feature_dim // heads
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        # x: [Batch, Seq_Len, Feature]
        B, T, C = x.shape
        residual = x
        
        # Kernel Feature Map: elu(x) + 1
        q = F.elu(self.q_proj(x).view(B, T, self.heads, self.head_dim)) + 1.0
        k = F.elu(self.k_proj(x).view(B, T, self.heads, self.head_dim)) + 1.0
        v = self.v_proj(x).view(B, T, self.heads, self.head_dim)
        
        # Linear Attention Calculation: Q * (K^T * V)
        kv = torch.einsum('bthd,bthe->bhde', k, v)
        z = torch.einsum('bthd,bhd->bth', q, k.sum(dim=1)).unsqueeze(-1)
        num = torch.einsum('bthd,bhde->bthe', q, kv)
        
        out = num / (z + 1e-6)
        out = out.reshape(B, T, C)
        out = self.out_proj(out)
        out = self.dropout(out)
        
        return self.norm(out + residual)

# ==========================================
# 6. 完整模型: ROEN_Final (Parallel Dual-Stream Version)
# ==========================================
class ROEN_Final(nn.Module):
    def __init__(self, node_in, edge_in, hidden, num_classes, seq_len=10, heads=8, dropout=0.3, max_cl_edges=2048):
        super(ROEN_Final, self).__init__()
        self.hidden = hidden
        self.seq_len = seq_len
        self.max_cl_edges = max_cl_edges
        
        # --- Encoders ---
        self.node_enc = nn.Linear(node_in, hidden)
        self.edge_enc = nn.Linear(edge_in, hidden)
        
        # --- Spatial Layers (Phase 1) ---
        self.num_layers = 2
        self.spatial_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.spatial_layers.append(nn.ModuleDict({
                'node_att': EdgeAugmentedAttention(hidden, hidden, hidden, heads, dropout, drop_path=0.1),
                'edge_upd': EdgeUpdaterModule(hidden, hidden, hidden, dropout)
            }))
            
        # --- Temporal Layers (Phase 2 & 3: Parallel) ---
        self.tpe = nn.Embedding(seq_len, hidden)
        
        # Stream A: Local (Inception)
        self.stream_local = TemporalInception1D(hidden, hidden) 
        
        # Stream B: Global (Linear Attention)
        self.stream_global = LinearTemporalAttention(hidden, heads, dropout)
        
        # Gating: Fusion Unit (New)
        self.gating = EntropyGatingUnit(hidden)
        
        # --- Contrastive Head (Phase 5) ---
        self.proj_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        
        # --- Classifier (Phase 4) ---
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 3, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, num_classes)
        )

    def forward(self, graphs):
        """
        前向传播逻辑：
        1. Spatial Evolution: 图卷积处理每一帧，同时使用 DropEdge 增强鲁棒性。
        2. Dynamic Alignment: 将不同时间步的节点特征对齐到全局 ID。
        3. Temporal Evolution (Parallel): 双流处理 + 门控融合。
        4. Readout & Classification: 结合节点和边特征进行最终分类。
        """
        spatial_node_feats = [] 
        spatial_edge_feats = [] 
        active_edge_indices = [] # 存储每帧 DropEdge 后保留的边索引
        edge_masks = []
        batch_global_ids = []
        
        # === Phase 1: Spatial Evolution ===
        for t in range(self.seq_len):
            data = graphs[t]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            batch = data.batch if hasattr(data, 'batch') else None
            
            # ID 对齐检查
            if hasattr(data, 'n_id'):
                batch_global_ids.append(data.n_id)
            elif hasattr(data, 'id'): 
                 batch_global_ids.append(data.id)
            else:
                batch_global_ids.append(torch.arange(x.size(0), device=x.device))

            if self.training:
                # [关键] DropEdge: 模拟网络中的链路不稳定或噪声
                edge_index, edge_mask = dropout_edge(edge_index, p=0.2, force_undirected=False)
                edge_attr = edge_attr[edge_mask]
                edge_masks.append(edge_mask)
            else:
                edge_masks.append(None)
            
            # 保存索引以供 Phase 4 使用，避免索引不匹配
            active_edge_indices.append(edge_index.clone())
            
            x = self.node_enc(x)
            edge_attr = self.edge_enc(edge_attr)
            
            for layer in self.spatial_layers:
                x = layer['node_att'](x, edge_index, edge_attr, batch)
                edge_attr = layer['edge_upd'](x, edge_index, edge_attr)
            
            spatial_node_feats.append(x)
            spatial_edge_feats.append(edge_attr)

        # === Phase 2: Dynamic Alignment (Sparse to Dense) ===
        all_ids = torch.cat(batch_global_ids)
        unique_ids, _ = torch.sort(torch.unique(all_ids))
        num_unique = len(unique_ids)
        device = unique_ids.device
        
        # [Num_Unique_Nodes, Seq_Len, Hidden]
        dense_stack = torch.zeros((num_unique, self.seq_len, self.hidden), device=device)
        
        for t in range(self.seq_len):
            indices = torch.searchsorted(unique_ids, batch_global_ids[t])
            dense_stack[indices, t, :] = spatial_node_feats[t]

        # === Phase 3: Temporal Evolution (Parallel Dual-Stream) ===
        
        # 3.1 添加时间位置编码
        time_indices = torch.arange(self.seq_len, device=device)
        t_emb = self.tpe(time_indices).unsqueeze(0) # [1, T, H]
        x_base = dense_stack + t_emb
        
        # 3.2 Stream A: Local (Inception)
        # Inception 需要 [Batch, Channel, Time]
        x_local_in = x_base.permute(0, 2, 1) 
        x_local = self.stream_local(x_local_in).permute(0, 2, 1) # 变回 [B, T, H]
        
        # 3.3 Stream B: Global (Linear Attention)
        # Attention 需要 [Batch, Time, Channel]
        x_global = self.stream_global(x_base)
        
        # 3.4 门控融合 (Gating Fusion)
        # dense_out = alpha * Local + (1-alpha) * Global + Residual
        dense_out, alpha_scores = self.gating(x_local, x_global, x_base)
        
        # === Phase 4 & 5: Readout & Contrastive ===
        batch_preds = []
        cl_loss = torch.tensor(0.0, device=device)
        
        for t in range(self.seq_len):
            # 1. 检索时序增强后的节点特征 (映射回 Sparse 格式)
            frame_ids = batch_global_ids[t]
            indices = torch.searchsorted(unique_ids, frame_ids)
            node_out_t = dense_out[indices, t, :] 
            
            # 2. 检索 Phase 1 保留的边索引
            curr_edge_index = active_edge_indices[t]
            src, dst = curr_edge_index[0], curr_edge_index[1]
            
            # 3. 边表示构建 = 边特征 + 源节点特征 + 宿节点特征
            edge_rep = torch.cat([
                spatial_edge_feats[t],
                node_out_t[src],
                node_out_t[dst]
            ], dim=1)
            
            # 4. 分类
            pred = self.classifier(edge_rep)
            batch_preds.append(pred)
            
            # 5. [Paper Logic] 对比学习
            # 仅在训练阶段，且只对中间帧计算（节省显存）
            if self.training and t == self.seq_len // 2:
                edge_feat_anchor = spatial_edge_feats[t]
                
                # 下采样 (避免 OOM)
                if edge_feat_anchor.size(0) > self.max_cl_edges:
                    perm = torch.randperm(edge_feat_anchor.size(0), device=device)[: self.max_cl_edges]
                    edge_feat_anchor = edge_feat_anchor[perm]

                # View 1: 原始特征的投影
                z1 = self.proj_head(edge_feat_anchor)
                
                # View 2: 高斯噪声扰动 (Gaussian Noise Perturbation)
                noise = torch.randn_like(edge_feat_anchor) * 0.1
                z2 = self.proj_head(edge_feat_anchor + noise)
                
                z1 = F.normalize(z1, dim=1)
                z2 = F.normalize(z2, dim=1)
                
                logits = torch.matmul(z1, z2.T) / 0.1
                labels = torch.arange(z1.size(0), device=device)
                cl_loss = F.cross_entropy(logits, labels)

        self._last_edge_masks = edge_masks
        # 可选：返回 alpha_scores 供可视化分析 (Case Study)
        # 这里为了保持接口兼容性，只返回 preds 和 loss
        return batch_preds, cl_loss
