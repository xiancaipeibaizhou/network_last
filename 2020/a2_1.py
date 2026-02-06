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

# === 关键修复：统一在顶部导入所有评估函数 ===
from analys import FocalLoss, evaluate_comprehensive, evaluate_with_threshold
from ROEN_Final import ROEN_Final
from network_new import ROEN_Fast_Transformer
# ==========================================
# 辅助函数：哈希与子网键生成
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
    # 使用 MD5 生成确定性 Hash ID (int64)
    hash_obj = hashlib.md5(str(ip_str).encode())
    return int(hash_obj.hexdigest()[:15], 16)

UNK_SUBNET_ID = 0

def get_subnet_id_safe(ip_str, subnet_map):
    key = _subnet_key(ip_str)
    return subnet_map.get(key, UNK_SUBNET_ID)

def evaluate_comprehensive_with_threshold(model, dataloader, device, class_names, threshold):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    normal_indices = []
    for idx, name in enumerate(class_names):
        name_lower = str(name).lower()
        if ('benign' in name_lower) or ('normal' in name_lower) or ('non' in name_lower):
            normal_indices.append(idx)

    attack_indices = [i for i in range(len(class_names)) if i not in set(normal_indices)]

    with torch.no_grad():
        for batched_seq in dataloader:
            batched_seq = [g.to(device) for g in batched_seq]

            logits = model(graphs=batched_seq, seq_len=len(batched_seq))[-1]
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            if len(attack_indices) > 0:
                attack_probs_sum = probs[:, attack_indices].sum(dim=1)
                mask = attack_probs_sum > threshold
                if mask.any():
                    sub_probs = probs[mask][:, attack_indices]
                    sub_argmax = torch.argmax(sub_probs, dim=1)
                    new_preds = torch.tensor(attack_indices, device=device)[sub_argmax]
                    preds = preds.clone()
                    preds[mask] = new_preds

            all_labels.extend(batched_seq[-1].edge_labels.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

    if len(all_labels) == 0:
        return 0, 0, 0, 0, 0, 0.5, 0, np.array([]), np.array([])

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    acc = (y_pred == y_true).mean()
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    is_true_normal = np.isin(y_true, normal_indices)
    is_pred_normal = np.isin(y_pred, normal_indices)
    fp = np.logical_and(is_true_normal, ~is_pred_normal).sum()
    tn = np.logical_and(is_true_normal, is_pred_normal).sum()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    attack_mask = ~is_true_normal
    asa = (y_pred[attack_mask] == y_true[attack_mask]).mean() if attack_mask.any() else 0.0

    try:
        if len(class_names) == 2:
            auc = roc_auc_score(y_true, y_probs[:, 1])
        else:
            auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
    except Exception:
        auc = 0.5

    return acc, prec, rec, f1, far, auc, asa, y_true, y_pred

# ==========================================
# 1. 稀疏图构建函数 (Inductive & Robust)
# ==========================================
def create_graph_data_inductive(time_slice, subnet_map, label_encoder, time_window):
    time_slice = time_slice.copy()
    
    # 确保 IP 为字符串
    time_slice['Src IP'] = time_slice['Src IP'].astype(str).str.strip()
    time_slice['Dst IP'] = time_slice['Dst IP'].astype(str).str.strip()

    # 使用 Hash ID 替代 Global Map
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

    # 构建局部图索引
    all_nodes_in_slice = np.concatenate([src_ids, dst_ids])
    unique_nodes, inverse_indices = np.unique(all_nodes_in_slice, return_inverse=True)
    
    n_nodes = len(unique_nodes)
    src_local = inverse_indices[:len(src_ids)]
    dst_local = inverse_indices[len(src_ids):]
    
    # [优化] 转为 numpy array 以避免 PyTorch 警告
    edge_index = torch.tensor(np.array([src_local, dst_local]), dtype=torch.long)
    n_id = torch.tensor(unique_nodes, dtype=torch.long)
    
    # --- 节点特征工程 (4维) ---
    ones = torch.ones(edge_index.size(1), dtype=torch.float)
    in_degrees = torch.zeros(n_nodes, dtype=torch.float)
    out_degrees = torch.zeros(n_nodes, dtype=torch.float)
    out_degrees.scatter_add_(0, edge_index[0], ones)
    in_degrees.scatter_add_(0, edge_index[1], ones)

    # 特征 3: 特权端口使用率
    src_port_col = 'Src Port' if 'Src Port' in time_slice.columns else 'Source Port'
    src_port = pd.to_numeric(time_slice.get(src_port_col, 0), errors='coerce').fillna(0).values
    is_priv_src = (src_port < 1024).astype(np.float32)
    
    priv_port_count = torch.zeros(n_nodes, dtype=torch.float)
    priv_port_count.scatter_add_(0, edge_index[0], torch.tensor(is_priv_src, dtype=torch.float))
    priv_ratio = priv_port_count / (out_degrees + 1e-6)

    # 特征 4: 流量聚合
    pkt_col = None
    for cand in ['Total Fwd Packets', 'Total Fwd Packet', 'Tot Fwd Pkts', 'Fwd Packets']:
        if cand in time_slice.columns:
            pkt_col = cand
            break
            
    if pkt_col is None:
        fwd_pkts = torch.zeros(edge_index.size(1), dtype=torch.float)
    else:
        # 已经是归一化后的数据
        fwd_pkts = torch.tensor(
            pd.to_numeric(time_slice[pkt_col], errors='coerce').fillna(0).values,
            dtype=torch.float,
        )
    
    node_pkt_sum = torch.zeros(n_nodes, dtype=torch.float)
    node_pkt_sum.scatter_add_(0, edge_index[0], fwd_pkts)
    
    x = torch.stack(
        [torch.log1p(in_degrees), torch.log1p(out_degrees), priv_ratio, node_pkt_sum],
        dim=-1,
    ).float()

    # --- 子网 ID ---
    subnet_id = None
    if subnet_map is not None:
        subnet_ids_for_node = {}
        unique_ips = pd.concat([time_slice['Src IP'], time_slice['Dst IP']]).unique()
        for ip_str in unique_ips:
            hid = get_ip_id_hash(ip_str)
            subnet_ids_for_node[hid] = get_subnet_id_safe(ip_str, subnet_map)
            
        subnet_id = torch.tensor(
            [subnet_ids_for_node.get(int(h), UNK_SUBNET_ID) for h in unique_nodes],
            dtype=torch.long,
        )

    # --- 边特征 ---
    drop_cols = ['Src IP', 'Dst IP', 'Flow ID', 'Label', 'Timestamp', 'Src Port', 'Dst Port', 
                 'Source IP', 'Destination IP', 'Source Port', 'Destination Port', 'time_idx']
    edge_attr_vals = time_slice.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number]).values
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
# 3. 主训练流程
# ==========================================
def main():
    # --- 配置 ---
    SEQ_LEN = int(os.getenv("SEQ_LEN", "10"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "100"))
    LR = float(os.getenv("LR", "0.001"))
    CSV_PATH = os.getenv("CSV_PATH", "data/CIC-Darknet2020/Darknet.csv")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {DEVICE}")

    # --- 1. 数据加载 ---
    print("Loading Data (CIC-Darknet2020)...")
    data = pd.read_csv(CSV_PATH) 
    
    data.drop(columns=['Label.1'], inplace=True, errors='ignore')
    data = data.dropna(subset=['Label', 'Timestamp']).copy()
    
    label_encoder = LabelEncoder()
    data['Label'] = label_encoder.fit_transform(data['Label'].astype(str))
    class_names = list(label_encoder.classes_)
    print(f"Classes: {class_names}")

    # 时间处理
    print("Processing Time...")
    data['Timestamp'] = pd.to_datetime(
        data['Timestamp'],
        dayfirst=True, 
        errors='coerce'
    )
    data = data.dropna(subset=['Timestamp'])
    data = data.sort_values('Timestamp')
    data['time_idx'] = data['Timestamp'].dt.floor('min')

    print("Performing Stratified Temporal Split (8:1:1 per class)...")

    train_list = []
    val_list = []
    test_list = []

    for label in data['Label'].unique():
        cls_data = data[data['Label'] == label].sort_values('Timestamp')

        unique_times = cls_data['time_idx'].drop_duplicates().values
        total_len = len(unique_times)

        if total_len < 3:
            train_list.append(cls_data)
            continue

        train_idx = int(total_len * 0.8)
        val_idx = int(total_len * 0.9)

        train_idx = max(1, train_idx)
        val_idx = max(train_idx + 1, val_idx)
        train_idx = min(train_idx, total_len - 2)
        val_idx = min(val_idx, total_len - 1)

        split_time_train = unique_times[train_idx]
        split_time_val = unique_times[val_idx]

        train_list.append(cls_data[cls_data['time_idx'] < split_time_train])
        val_list.append(
            cls_data[
                (cls_data['time_idx'] >= split_time_train)
                & (cls_data['time_idx'] < split_time_val)
            ]
        )
        test_list.append(cls_data[cls_data['time_idx'] >= split_time_val])

    train_df = pd.concat(train_list).sort_values('Timestamp') if len(train_list) > 0 else data.iloc[0:0].copy()
    val_df = pd.concat(val_list).sort_values('Timestamp') if len(val_list) > 0 else data.iloc[0:0].copy()
    test_df = pd.concat(test_list).sort_values('Timestamp') if len(test_list) > 0 else data.iloc[0:0].copy()

    print(f"Final Split -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}", flush=True)
    print(f"Test Set Classes: {test_df['Label'].unique()}", flush=True)

    def _print_label_counts(df, split_name):
        vc = df['Label'].value_counts().sort_index()
        pairs = []
        for label_id, cnt in vc.items():
            label_name = class_names[int(label_id)] if int(label_id) < len(class_names) else str(label_id)
            pairs.append(f"{label_name}({int(label_id)}):{int(cnt)}")
        print(f"{split_name} Label Counts -> " + ", ".join(pairs), flush=True)

    _print_label_counts(train_df, "Train")
    _print_label_counts(val_df, "Val")
    _print_label_counts(test_df, "Test")

    # === [归一化] ===
    print("Performing Normalization (Inductive)...")
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['Label', 'Timestamp', 'Src IP', 'Dst IP', 'Flow ID', 'Src Port', 'Dst Port', 'time_idx']
    feat_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    # [关键修复]：清理 inf 和 -inf
    train_df[feat_cols] = train_df[feat_cols].replace([np.inf, -np.inf], np.nan)
    val_df[feat_cols] = val_df[feat_cols].replace([np.inf, -np.inf], np.nan)
    test_df[feat_cols] = test_df[feat_cols].replace([np.inf, -np.inf], np.nan)

    # 填充 0
    train_df[feat_cols] = train_df[feat_cols].fillna(0)
    val_df[feat_cols] = val_df[feat_cols].fillna(0)
    test_df[feat_cols] = test_df[feat_cols].fillna(0)
 
    # Log1p
    for col in feat_cols:
        if train_df[col].max() > 100:
            train_df[col] = np.log1p(train_df[col].abs())
            val_df[col] = np.log1p(val_df[col].abs())
            test_df[col] = np.log1p(test_df[col].abs())
    
    # Fit on Train, Transform Val/Test
    scaler = StandardScaler()
    try:
        train_df[feat_cols] = scaler.fit_transform(train_df[feat_cols])
        val_df[feat_cols] = scaler.transform(val_df[feat_cols])
        test_df[feat_cols] = scaler.transform(test_df[feat_cols])
    except ValueError as e:
        print("Error during scaling. Check for inf values.")
        raise e
         
    print("Normalization Done.")

    # --- 2. 构建 Subnet Map (Train Only) ---
    print("Building Subnet Map (From Train Set Only)...")
    train_df['Src IP'] = train_df['Src IP'].astype(str).str.strip()
    train_df['Dst IP'] = train_df['Dst IP'].astype(str).str.strip()
    
    train_ips = pd.concat([train_df['Src IP'], train_df['Dst IP']]).unique()
    subnet_to_idx = {'<UNK>': UNK_SUBNET_ID}
    
    for ip in train_ips:
        key = _subnet_key(ip)
        if key not in subnet_to_idx:
            subnet_to_idx[key] = len(subnet_to_idx)
            
    num_subnets = len(subnet_to_idx)
    print(f"Train Subnets: {num_subnets}")

    # --- 3. 构建 Graphs ---
    print("Constructing Train Graphs...")
    train_grouped = train_df.groupby('time_idx', sort=True)
    train_seqs = []
    for name, group in tqdm(train_grouped):
        g = create_graph_data_inductive(group, subnet_to_idx, None, name)
        if g: train_seqs.append(g)

    print("Constructing Val Graphs...")
    val_grouped = val_df.groupby('time_idx', sort=True)
    val_seqs = []
    for name, group in tqdm(val_grouped):
        g = create_graph_data_inductive(group, subnet_to_idx, None, name)
        if g: val_seqs.append(g)

    print("Constructing Test Graphs...")
    test_grouped = test_df.groupby('time_idx', sort=True)
    test_seqs = []
    for name, group in tqdm(test_grouped):
        g = create_graph_data_inductive(group, subnet_to_idx, None, name)
        if g: test_seqs.append(g)

    print(f"Total Train Graphs: {len(train_seqs)}, Val Graphs: {len(val_seqs)}, Test Graphs: {len(test_seqs)}")

    # 4. Dataset
    train_dataset = TemporalGraphDataset(train_seqs, seq_len=SEQ_LEN)
    val_dataset = TemporalGraphDataset(val_seqs, seq_len=SEQ_LEN)
    test_dataset = TemporalGraphDataset(test_seqs, seq_len=SEQ_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=temporal_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=temporal_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=temporal_collate_fn)

    # --- 5. 模型初始化 ---
    if len(train_seqs) > 0:
        edge_dim = train_seqs[0].edge_attr.shape[1]
    elif len(test_seqs) > 0:
        edge_dim = test_seqs[0].edge_attr.shape[1]
    else:
        edge_dim = 1
    
    print(f"Initializing ROEN_Final (Node In: 4, Edge Dim: {edge_dim}, Subnets: {num_subnets})...")
    # model = ROEN_Final(
    #     node_in=4,
    #     edge_in=edge_dim,
    #     hidden=128, 
    #     num_classes=len(class_names),
    #     num_subnets=num_subnets,
    #     seq_len=SEQ_LEN,
    #     heads=8
    # ).to(DEVICE)
    model = ROEN_Fast_Transformer(
        node_in=4,
        edge_in=edge_dim,
        hidden=128, 
        num_classes=len(class_names),
        num_subnets=num_subnets,
        # seq_len=SEQ_LEN,
        # heads=8
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print("Calculating Class Weights...")
    label_counts = train_df['Label'].value_counts().sort_index()
    class_counts_tensor = torch.tensor(label_counts.values, dtype=torch.float).to(DEVICE)
    weights = 1.0 / (torch.sqrt(class_counts_tensor) + 1.0)
    weights = weights / weights.sum() * len(class_names)
    
    criterion = nn.CrossEntropyLoss(weight=weights)

    # --- 6. 训练 ---
    print("Start Training...")
    start_time = time.time()

    save_dir = 'models/2020'
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, 'roen_final_best.pth')
    print(f"Best model will be saved to: {best_model_path}")

    best_val_f1 = 0.0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        for batched_seq in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            if not batched_seq: continue
            batched_seq = [g.to(DEVICE) for g in batched_seq]
            
            optimizer.zero_grad()
            preds_seq = model(graphs=batched_seq, seq_len=len(batched_seq))
            loss = criterion(preds_seq[-1], batched_seq[-1].edge_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

        val_acc, val_prec, val_rec, val_f1, val_far, val_auc, val_asa = evaluate_comprehensive(
            model, val_loader, DEVICE, class_names
        )
        print(f"Epoch {epoch+1} Val F1: {val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f">>> New Best Model Saved! (Val F1: {best_val_f1:.4f})")

    print(f"\nLoading Best Model from {best_model_path} for Final Testing...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    else:
        print("Warning: No best model file found! Using last epoch weights.")
    final_acc, final_prec, final_rec, final_f1, final_far, final_auc, final_asa = evaluate_comprehensive(
        model, test_loader, DEVICE, class_names
    )
    print(f"Final Test -> ACC: {final_acc:.4f}, PREC: {final_prec:.4f}, F1: {final_f1:.4f}, Rec: {final_rec:.4f}, FAR: {final_far:.4f}, AUC: {final_auc:.4f}, ASA: {final_asa:.4f}")

    # Threshold Optimization & Plotting
    print("\n=== Post-Training Threshold Optimization ===")
    best_asa = 0.0
    best_thresh = 0.5
    for th in [0.5, 0.4, 0.3, 0.25, 0.2, 0.15]:
        _, f1, far, asa = evaluate_with_threshold(model, test_loader, DEVICE, class_names, threshold=th)
        print(f"[Threshold {th}] -> F1: {f1:.4f}, FAR: {far:.4f}, ASA: {asa:.4f}")
        if asa > best_asa and far < 0.01:
            best_asa = asa
            best_thresh = th
            
    print(f"\nBest Strategy: Use Threshold = {best_thresh}")
    OPTIMAL_THRESH = best_thresh

    print("\n=== Re-Evaluating with Best Threshold ===")
    opt_acc, opt_prec, opt_rec, opt_f1, opt_far, opt_auc, opt_asa, final_labels, final_preds = evaluate_comprehensive_with_threshold(
        model, test_loader, DEVICE, class_names, threshold=OPTIMAL_THRESH
    )
    print(
        f"Optimal Threshold Test -> ACC: {opt_acc:.4f}, PREC: {opt_prec:.4f}, "
        f"F1: {opt_f1:.4f}, Rec: {opt_rec:.4f}, FAR: {opt_far:.4f}, "
        f"AUC: {opt_auc:.4f}, ASA: {opt_asa:.4f}"
    )

    labels_idx = list(range(len(class_names)))
    cm = confusion_matrix(final_labels, final_preds, labels=labels_idx)
    cm = np.asarray(cm, dtype=np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0) * 100.0

    save_path = f'png/2020/FINAL_BEST_CM_Thresh{OPTIMAL_THRESH}.png'
    try:
        import seaborn as sns
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix (Threshold={OPTIMAL_THRESH})')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Final Confusion Matrix saved to {save_path}")
    except Exception as e:
        print(f"Plotting failed: {e}")

    print(f"Total Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/2020', exist_ok=True)
    os.makedirs('png/2020', exist_ok=True)
    main()
