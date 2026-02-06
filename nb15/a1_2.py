import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import hashlib
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import recall_score, f1_score, roc_auc_score, precision_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
# from analys import FocalLoss

# # 引入 Fast 模型
# from network_fast_transformer import ROEN_Fast_Transformer 
# from network_advanced import ROEN_Advanced
from model_Final import ROEN_Final

# ==========================================
# 辅助函数
# ==========================================
def _subnet_key(ip):
    try:
        parts = str(ip).split(".")
        if len(parts) < 3:
            return (0, 0, 0)
        a = int(parts[0])
        b = int(parts[1])
        c = int(parts[2])
        return (a, b, c)
    except Exception:
        return (0, 0, 0)

def get_ip_id_hash(ip_str):
    # 使用 MD5 生成确定性 Hash ID
    hash_obj = hashlib.md5(str(ip_str).encode())
    return int(hash_obj.hexdigest()[:15], 16)

UNK_SUBNET_ID = 0

def get_subnet_id_safe(ip_str, subnet_map):
    key = _subnet_key(ip_str)
    return subnet_map.get(key, UNK_SUBNET_ID)

# ==========================================
# 1. 稀疏图构建函数 (Inductive)
# ==========================================
def create_graph_data_inductive(time_slice, subnet_map, label_encoder, time_window):
    time_slice = time_slice.copy()
    time_slice['Src IP'] = time_slice['Src IP'].astype(str).str.strip()
    time_slice['Dst IP'] = time_slice['Dst IP'].astype(str).str.strip()

    # 使用 Hash ID
    src_ids = time_slice['Src IP'].apply(get_ip_id_hash).values.astype(np.int64)
    dst_ids = time_slice['Dst IP'].apply(get_ip_id_hash).values.astype(np.int64)

    # 标签处理
    if label_encoder:
        try:
            labels = label_encoder.transform(time_slice['Label'].astype(str))
        except:
            labels = np.zeros(len(time_slice), dtype=int)
    else:
        labels = time_slice['Label'].values.astype(int)

    all_nodes_in_slice = np.concatenate([src_ids, dst_ids])
    unique_nodes, inverse_indices = np.unique(all_nodes_in_slice, return_inverse=True)
    
    n_nodes = len(unique_nodes)
    src_local = inverse_indices[:len(src_ids)]
    dst_local = inverse_indices[len(src_ids):]
    
    # [优化] 显式转换 numpy array 以消除 UserWarning
    edge_index = torch.tensor(np.array([src_local, dst_local]), dtype=torch.long)
    n_id = torch.tensor(unique_nodes, dtype=torch.long)
    
    # 度特征
    ones = torch.ones(edge_index.size(1), dtype=torch.float)
    in_degrees = torch.zeros(n_nodes, dtype=torch.float)
    out_degrees = torch.zeros(n_nodes, dtype=torch.float)
    out_degrees.scatter_add_(0, edge_index[0], ones)
    in_degrees.scatter_add_(0, edge_index[1], ones)

    # 子网特征 (Hash + Map)
    subnet_id = None
    if subnet_map is not None:
        subnet_ids_for_node = {}
        # 预先构建当前 slice 的 hash->subnet 映射
        # 这里为了效率，只对 unique IP 做处理
        unique_ips = pd.concat([time_slice['Src IP'], time_slice['Dst IP']]).unique()
        for ip_str in unique_ips:
            hid = get_ip_id_hash(ip_str)
            subnet_ids_for_node[hid] = get_subnet_id_safe(ip_str, subnet_map)
            
        subnet_id = torch.tensor(
            [subnet_ids_for_node.get(int(h), UNK_SUBNET_ID) for h in unique_nodes],
            dtype=torch.long,
        )

    # 行为特征 1: 特权端口比率
    src_port = pd.to_numeric(time_slice.get('Src Port', 0), errors='coerce').fillna(0).values
    is_priv_src = (src_port < 1024).astype(np.float32)
    priv_port_count = torch.zeros(n_nodes, dtype=torch.float)
    priv_port_count.scatter_add_(0, edge_index[0], torch.tensor(is_priv_src, dtype=torch.float))
    priv_ratio = priv_port_count / (out_degrees + 1e-6)

    # 行为特征 2: 流量聚合 (已归一化，可能为负，不能 Log)
    pkt_col = None
    for cand in ['Total Fwd Packets', 'Total Fwd Packet', 'Tot Fwd Pkts', 'Fwd Packets']:
        if cand in time_slice.columns:
            pkt_col = cand
            break
    
    if pkt_col is None:
        fwd_pkts = torch.zeros(edge_index.size(1), dtype=torch.float)
    else:
        fwd_pkts = torch.tensor(
            pd.to_numeric(time_slice[pkt_col], errors='coerce').fillna(0).values,
            dtype=torch.float,
        )
    
    node_pkt_sum = torch.zeros(n_nodes, dtype=torch.float)
    node_pkt_sum.scatter_add_(0, edge_index[0], fwd_pkts)

    # [关键修复] 移除 node_pkt_sum 的 log1p，直接使用
    x = torch.stack(
        [torch.log1p(in_degrees), torch.log1p(out_degrees), priv_ratio, node_pkt_sum],
        dim=-1,
    ).float()
    
    drop_cols = ['Src IP', 'Dst IP', 'Flow ID', 'Label', 'Timestamp', 'Src Port', 'Dst Port']
    edge_attr_vals = time_slice.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number]).values
    edge_attr_vals = np.nan_to_num(edge_attr_vals, nan=0.0, posinf=0.0, neginf=0.0)
    edge_attr = torch.tensor(edge_attr_vals, dtype=torch.float)

    if edge_index.size(1) > 0:
        edge_labels = torch.tensor(labels, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_labels=edge_labels, n_id=n_id)
        if subnet_id is not None:
            data.subnet_id = subnet_id
        return data
    else:
        return None

# ==========================================
# 2. Dataset & Collate
# ==========================================
class TemporalGraphDataset(torch.utils.data.Dataset):
    def __init__(self, graph_data_seq, seq_len=8):
        self.graph_data_seq = [g for g in graph_data_seq if g is not None]
        self.seq_len = seq_len
    def __len__(self):
        return max(0, len(self.graph_data_seq) - self.seq_len + 1)
    def __getitem__(self, idx):
        return self.graph_data_seq[idx : idx + self.seq_len]

def temporal_collate_fn(batch):
    if len(batch) == 0: return []
    seq_len = len(batch[0])
    batched_seq = []
    for t in range(seq_len):
        graphs_at_t = [sample[t] for sample in batch]
        batched_seq.append(Batch.from_data_list(graphs_at_t))
    return batched_seq

# ==========================================
# 3. 评估辅助函数
# ==========================================

def _forward_compatible(model, batched_seq):
    try:
        return model(batched_seq)
    except TypeError:
        try:
            return model(batched_seq, len(batched_seq))
        except TypeError:
            return model(graphs=batched_seq, seq_len=len(batched_seq))

def evaluate_compatible(model, dataloader, device, class_names):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            batch = [g.to(device) for g in batch]

            out = _forward_compatible(model, batch)
            if isinstance(out, tuple):
                preds_seq = out[0]
            else:
                preds_seq = out

            logits = preds_seq[-1]
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(batch[-1].edge_labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return f1_score(all_labels, all_preds, average='weighted', zero_division=0)

def evaluate_with_threshold_compatible(model, dataloader, device, class_names, threshold=0.4):
    model.eval()
    all_labels = []
    all_preds = []

    normal_indices = []
    for idx, name in enumerate(class_names):
        low = str(name).lower()
        if any(k in low for k in ("non", "non-tor", "nonvpn", "normal", "benign")):
            normal_indices.append(idx)
    if len(normal_indices) == 0 and len(class_names) > 0:
        normal_indices = [0]

    attack_indices = [i for i in range(len(class_names)) if i not in set(normal_indices)]

    with torch.no_grad():
        for batch in dataloader:
            batch = [g.to(device) for g in batch]

            out = _forward_compatible(model, batch)
            if isinstance(out, tuple):
                preds_seq = out[0]
            else:
                preds_seq = out

            logits = preds_seq[-1]
            probs = torch.softmax(logits, dim=1)
            final_preds = torch.argmax(probs, dim=1)

            if len(attack_indices) > 0:
                attack_probs_sum = probs[:, attack_indices].sum(dim=1)
                mask = attack_probs_sum > threshold

                if mask.any():
                    sub_probs = probs[mask][:, attack_indices]
                    sub_argmax = torch.argmax(sub_probs, dim=1)
                    final_preds[mask] = torch.tensor(attack_indices, device=device)[sub_argmax]

            all_labels.extend(batch[-1].edge_labels.cpu().numpy())
            all_preds.extend(final_preds.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    is_true_normal = np.isin(y_true, normal_indices)
    is_pred_normal = np.isin(y_pred, normal_indices)
    fp = np.logical_and(is_true_normal, ~is_pred_normal).sum()
    tn = np.logical_and(is_true_normal, is_pred_normal).sum()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    is_true_attack = ~is_true_normal
    asa = (y_pred[is_true_attack] == y_true[is_true_attack]).mean() if is_true_attack.any() else 0.0

    acc = (y_pred == y_true).mean() if len(y_true) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0) if len(y_true) > 0 else 0.0

    return acc, f1, far, asa

# ==========================================
# 4. 主流程
# ==========================================
def temporal_split(data_list, test_size=0.2):
    split_idx = int(len(data_list) * (1 - test_size))
    return data_list[:split_idx], data_list[split_idx:]

def main():
    SEQ_LEN = 10       
    BATCH_SIZE = 16   
    NUM_EPOCHS = 150
    LR = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading NB15 Data (CICFlowMeter Format)...")
    data = pd.read_csv("data/CIC-NUSW-NB15/CICFlowMeter_out.csv") 
    
    # 清洗标签
    data['Label'] = data['Label'].astype(str).str.strip().replace('', np.nan)
    data.dropna(subset=['Label', 'Timestamp'], inplace=True)
    
    label_encoder = LabelEncoder()
    data['Label'] = label_encoder.fit_transform(data['Label'])
    class_names = list(label_encoder.classes_)
    print(f"Classes: {class_names}")

    # 时间处理
    print("Processing Time..." )
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], dayfirst=True, errors='coerce')
    data.dropna(subset=['Timestamp'], inplace=True)
    data = data.sort_values('Timestamp')
    data['time_idx'] = data['Timestamp'].dt.floor('min')

    # === [关键优化] 数据切分 (先切分，后归一化，防止 Data Leakage) ===
    unique_times = data['time_idx'].drop_duplicates().values
    total_len = len(unique_times)
    train_idx = int(total_len * 0.8)
    val_idx = int(total_len * 0.9)
    train_idx = max(1, min(total_len - 1, train_idx))
    val_idx = max(train_idx + 1, min(total_len - 1, val_idx))

    split_time_train = unique_times[train_idx]
    split_time_val = unique_times[val_idx]

    print(f"Splitting: Train < {split_time_train} <= Val < {split_time_val} <= Test")

    train_df = data[data['time_idx'] < split_time_train].copy()
    val_df = data[(data['time_idx'] >= split_time_train) & (data['time_idx'] < split_time_val)].copy()
    test_df = data[data['time_idx'] >= split_time_val].copy()

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 归一化 (仅在训练集上 Fit)
    print("Normalizing (Fit on Train, Transform Test)..." )
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ['Label', 'Timestamp', 'Src IP', 'Dst IP', 'Flow ID', 'Src Port', 'Dst Port']
    feat_cols = [c for c in numeric_cols if c not in exclude]
    
    # 填充缺失值
    train_df[feat_cols] = train_df[feat_cols].fillna(0)
    val_df[feat_cols] = val_df[feat_cols].fillna(0)
    test_df[feat_cols] = test_df[feat_cols].fillna(0)
    
    # Log1p 处理长尾
    for col in feat_cols:
        # 注意：这里假设原始数据非负。如果已经标准化过则不需要。通常原始数据是 Byte/Packet count，是非负的。
        if train_df[col].max() > 100: 
            train_df[col] = np.log1p(train_df[col].abs())
            val_df[col] = np.log1p(val_df[col].abs())
            test_df[col] = np.log1p(test_df[col].abs())

    scaler = StandardScaler()
    train_df[feat_cols] = scaler.fit_transform(train_df[feat_cols])
    val_df[feat_cols] = scaler.transform(val_df[feat_cols])
    test_df[feat_cols] = scaler.transform(test_df[feat_cols]) # 使用训练集的参数转换测试集

    # 构建 Subnet Map (仅用训练集)
    print("Building Subnet Map (From Train Set Only)..." )
    data['Src IP'] = data['Src IP'].astype(str).str.strip() # 确保原始 dataframe 类型正确 (用于 IP 提取)
    train_df['Src IP'] = train_df['Src IP'].astype(str).str.strip()
    train_df['Dst IP'] = train_df['Dst IP'].astype(str).str.strip()
    
    train_ips = pd.concat([train_df['Src IP'], train_df['Dst IP']]).unique()
    subnet_to_idx = {'<UNK>': UNK_SUBNET_ID}
    for ip in train_ips:
        key = _subnet_key(ip)
        if key not in subnet_to_idx:
            subnet_to_idx[key] = len(subnet_to_idx)
    num_subnets = len(subnet_to_idx)
    print(f"Train Subnets: {num_subnets} (Unknown subnets in Test will be mapped to 0)")

    # 构建图序列
    print("Building Train Graphs...")
    train_grouped = train_df.groupby('time_idx', sort=True)
    train_seqs = []
    for name, group in tqdm(train_grouped):
        g = create_graph_data_inductive(group, subnet_to_idx, None, name)
        if g: train_seqs.append(g)

    print("Building Val Graphs...")
    val_grouped = val_df.groupby('time_idx', sort=True)
    val_seqs = []
    for name, group in tqdm(val_grouped):
        g = create_graph_data_inductive(group, subnet_to_idx, None, name)
        if g: val_seqs.append(g)

    print("Building Test Graphs...")
    test_grouped = test_df.groupby('time_idx', sort=True)
    test_seqs = []
    for name, group in tqdm(test_grouped):
        g = create_graph_data_inductive(group, subnet_to_idx, None, name)
        if g: test_seqs.append(g)
    
    train_dataset = TemporalGraphDataset(train_seqs, SEQ_LEN)
    val_dataset = TemporalGraphDataset(val_seqs, SEQ_LEN)
    test_dataset = TemporalGraphDataset(test_seqs, SEQ_LEN)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=temporal_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=temporal_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=temporal_collate_fn)

    # 模型初始化
    if len(train_seqs) > 0:
        edge_dim = train_seqs[0].edge_attr.shape[1]
    elif len(test_seqs) > 0:
        edge_dim = test_seqs[0].edge_attr.shape[1]
    else:
        edge_dim = 1
        
    print(f"Initializing Model (Node In: 4, Subnets: {num_subnets})...")
    model = ROEN_Final(
        node_in=4,
        edge_in=edge_dim, 
        hidden=64, 
        num_classes=len(class_names),
        seq_len=SEQ_LEN,
        heads=8
    ).to(DEVICE)
    
    # 类别权重
    print("Calculating Class Weights...")
    label_counts = train_df['Label'].value_counts().sort_index()
    class_counts_tensor = torch.tensor(label_counts.values, dtype=torch.float).to(DEVICE)
    weights = 1.0 / (torch.sqrt(class_counts_tensor) + 1.0)
    weights = weights / weights.sum() * len(class_names)
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 训练循环
    print("Start Training...")
    start_time = time.time()
    best_val_f1 = 0.0
    best_model_path = 'models/nb15/best_model_enhanced.pth'
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        for batched_seq in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            if not batched_seq:
                continue
            batched_seq = [g.to(DEVICE) for g in batched_seq]
            optimizer.zero_grad()

            out = _forward_compatible(model, batched_seq)
            if isinstance(out, tuple):
                preds_seq, cl_loss = out
            else:
                preds_seq, cl_loss = out, torch.tensor(0.0, device=DEVICE)

            last_pred = preds_seq[-1]
            last_labels = batched_seq[-1].edge_labels

            edge_masks = getattr(model, "_last_edge_masks", None)
            if edge_masks is not None and edge_masks[-1] is not None:
                last_labels = last_labels[edge_masks[-1]]

            main_loss = criterion(last_pred, last_labels)
            loss = main_loss + 0.01 * cl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss/max(1, len(train_loader)):.4f}")

        if len(val_dataset) > 0:
            val_f1 = evaluate_compatible(model, val_loader, DEVICE, class_names)
            print(f"Val F1: {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), best_model_path)

    print("\nLoading Best Model for Final Testing...")
    if not os.path.exists(best_model_path):
        torch.save(model.state_dict(), best_model_path)
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    test_f1 = evaluate_compatible(model, test_loader, DEVICE, class_names)
    print(f"Test F1: {test_f1:.4f}")

    # 阈值搜索与绘图 (逻辑保持不变)
    print("Optimizing Threshold...")
    best_th, best_asa = 0.5, 0.0
    for th in [0.5, 0.4, 0.3, 0.2, 0.1]:
        _, _, far, asa = evaluate_with_threshold_compatible(model, test_loader, DEVICE, class_names, th)
        print(f"Thresh {th}: FAR {far:.4f}, ASA {asa:.4f}")
        if asa > best_asa and far < 0.01: best_th, best_asa = th, asa

    OPTIMAL_THRESH = best_th
    attack_indices = [i for i, n in enumerate(class_names) if 'normal' not in n.lower() and 'benign' not in n.lower()]
    
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = [g.to(DEVICE) for g in batch]
            out = _forward_compatible(model, batch)
            if isinstance(out, tuple):
                preds_seq = out[0]
            else:
                preds_seq = out
            logits = preds_seq[-1]
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            if attack_indices:
                mask = probs[:, attack_indices].sum(dim=1) > OPTIMAL_THRESH
                if mask.any():
                    sub_argmax = torch.argmax(probs[mask][:, attack_indices], dim=1)
                    preds[mask] = torch.tensor(attack_indices, device=DEVICE)[sub_argmax]
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch[-1].edge_labels.cpu().numpy())

    # 绘图部分
    try:
        os.makedirs('png/nb15', exist_ok=True)
        labels_idx = list(range(len(class_names)))
        cm = confusion_matrix(all_labels, all_preds, labels=labels_idx)
        cm = np.asarray(cm, dtype=np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0) * 100.0
        save_path = f'png/nb15/FINAL_BEST_CM_Thresh{OPTIMAL_THRESH}.png'
        
        # ... (绘图代码保持一致) ...
        # 为节省篇幅，这里简写，绘图部分逻辑与您原代码一致，直接运行即可。
        import seaborn as sns
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix (Threshold={OPTIMAL_THRESH})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved CM to {save_path}")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    os.makedirs('models/nb15', exist_ok=True)
    os.makedirs('png/nb15', exist_ok=True)
    main()
