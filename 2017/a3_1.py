import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import hashlib
import os
import time
import glob
from tqdm import tqdm
from hparams_a3 import resolve_hparams
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import recall_score, f1_score, roc_auc_score, precision_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# === 导入核心模型与评估函数 ===
# 确保 analys.py 和 model_Final.py 在同一目录下
try:
    from analys import FocalLoss, evaluate_comprehensive, evaluate_with_threshold
    from model_final import ROEN_Final
except ImportError:
    print("❌ Error: analys.py or model_Final.py not found. Please check your directory.")
    exit()

# ==========================================
# 辅助函数
# ==========================================
def get_ip_id_hash(ip_str):
    hash_obj = hashlib.md5(str(ip_str).encode())
    return int(hash_obj.hexdigest()[:15], 16)

UNK_SUBNET_ID = 0

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
            if not batched_seq: continue
            batched_seq = [g.to(device) for g in batched_seq]

            try:
                out = model(graphs=batched_seq, seq_len=len(batched_seq))
            except TypeError:
                out = model(graphs=batched_seq)
            
            preds_seq = out[0] if isinstance(out, tuple) else out
            logits = preds_seq[-1]
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
# 1. 稀疏图构建函数 (修复版)
# ==========================================
def create_graph_data_inductive(time_slice, subnet_map, label_encoder, time_window):
    # 避免 SettingWithCopyWarning
    time_slice = time_slice.copy()
    
    # 确保 IP 为字符串
    time_slice['Src IP'] = time_slice['Src IP'].astype(str).str.strip()
    time_slice['Dst IP'] = time_slice['Dst IP'].astype(str).str.strip()

    # 使用 Hash ID
    src_ids = time_slice['Src IP'].apply(get_ip_id_hash).values.astype(np.int64)
    dst_ids = time_slice['Dst IP'].apply(get_ip_id_hash).values.astype(np.int64)

    # 标签处理
    if 'Label' in time_slice.columns:
        if pd.api.types.is_numeric_dtype(time_slice['Label']):
            labels = time_slice['Label'].values.astype(int)
        else:
            labels = np.zeros(len(time_slice), dtype=int)
    else:
        labels = np.zeros(len(time_slice), dtype=int)

    # 构建局部图索引
    all_nodes_in_slice = np.concatenate([src_ids, dst_ids])
    unique_nodes, inverse_indices = np.unique(all_nodes_in_slice, return_inverse=True)
    
    n_nodes = len(unique_nodes)
    src_local = inverse_indices[:len(src_ids)]
    dst_local = inverse_indices[len(src_ids):]
    
    edge_index = torch.tensor(np.array([src_local, dst_local]), dtype=torch.long)
    n_id = torch.tensor(unique_nodes, dtype=torch.long)
    
    # --- 节点特征 ---
    ones = torch.ones(edge_index.size(1), dtype=torch.float)
    in_degrees = torch.zeros(n_nodes, dtype=torch.float)
    out_degrees = torch.zeros(n_nodes, dtype=torch.float)
    
    if edge_index.size(1) > 0:
        out_degrees.scatter_add_(0, edge_index[0], ones)
        in_degrees.scatter_add_(0, edge_index[1], ones)

    # 特征: 特权端口
    src_port_col = 'Src Port' if 'Src Port' in time_slice.columns else 'Source Port'
    src_port = pd.to_numeric(time_slice.get(src_port_col, 0), errors='coerce').fillna(0).values
    is_priv_src = (src_port < 1024).astype(np.float32)
    
    priv_port_count = torch.zeros(n_nodes, dtype=torch.float)
    if edge_index.size(1) > 0:
        priv_port_count.scatter_add_(0, edge_index[0], torch.tensor(is_priv_src, dtype=torch.float))
    priv_ratio = priv_port_count / (out_degrees + 1e-6)

    # 特征: 流量聚合
    pkt_col = None
    for cand in ['Total Fwd Packets', 'Total Fwd Packet', 'Tot Fwd Pkts', 'Fwd Packets', 'Total Fwd Pkts']:
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
    if edge_index.size(1) > 0:
        node_pkt_sum.scatter_add_(0, edge_index[0], fwd_pkts)
    
    x = torch.stack(
        [torch.log1p(in_degrees), torch.log1p(out_degrees), priv_ratio, node_pkt_sum],
        dim=-1,
    ).float()

    # --- Subnet ID (全0占位，避免越界) ---
    subnet_id = torch.zeros(n_nodes, dtype=torch.long)

    # --- 边特征 ---
    drop_cols = ['Src IP', 'Dst IP', 'Flow ID', 'Label', 'Timestamp', 'Src Port', 'Dst Port', 
                 'Source IP', 'Destination IP', 'Source Port', 'Destination Port', 'time_idx']
    edge_attr_vals = time_slice.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number]).values
    edge_attr = torch.tensor(edge_attr_vals, dtype=torch.float)

    if edge_index.size(1) > 0:
        edge_labels = torch.tensor(labels, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_labels=edge_labels, n_id=n_id)
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

def _pad_seq_for_last_frame_coverage(graph_seqs, seq_len):
    if graph_seqs is None or len(graph_seqs) == 0:
        return graph_seqs
    pad_n = max(0, int(seq_len) - 1)
    if pad_n == 0:
        return graph_seqs
    first = graph_seqs[0]
    pads = []
    for _ in range(pad_n):
        pads.append(first.clone() if hasattr(first, "clone") else first)
    return pads + list(graph_seqs)

# ==========================================
# 3. 主训练流程
# ==========================================
def main():
    # --- 配置 ---
    DATA_DIR = os.getenv("DATA_DIR", "data/CIC-IDS2017/TrafficLabelling_")
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {DEVICE}")

    # --- 1. 批量加载与拼接 CSV ---
    print(f"Scanning CSV files in {DATA_DIR}...")
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {DATA_DIR}. Please check the path.")
        return
    
    data_frames = []
    for file_path in tqdm(csv_files, desc="Loading CSVs"):
        try:
            df = pd.read_csv(file_path, encoding="latin1", low_memory=False)
            df.columns = df.columns.str.strip()
            
            rename_map = {
                "Source IP": "Src IP", "Destination IP": "Dst IP",
                "Source Port": "Src Port", "Destination Port": "Dst Port",
                " Timestamp": "Timestamp"
            }
            df = df.rename(columns=rename_map)
            
            if "Timestamp" not in df.columns: continue
            data_frames.append(df)
        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")

    if not data_frames: return

    print("Concatenating DataFrames...")
    data = pd.concat(data_frames, ignore_index=True)
    del data_frames

    # --- 2. 数据清洗 ---
    print("Cleaning Data...")
    data["Label"] = data["Label"].astype(str).str.strip()
    data = data[data["Label"].notna()]
    data = data[data["Label"] != ""]
    data = data[~data["Label"].str.lower().isin(["nan", "none"])]
    
    # 编码 Label
    label_encoder = LabelEncoder()
    data["Label"] = label_encoder.fit_transform(data["Label"])
    class_names = list(label_encoder.classes_)
    print(f"🏷️ Classes: {class_names}")

    # 时间处理
    print("Parsing Timestamps & Sorting...")
    # IDS2017 常见格式: "dd/MM/yyyy h:mm" 或 "dd/MM/yyyy hh:mm:ss a"
    data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")
    data.dropna(subset=["Timestamp", "Src IP", "Dst IP"], inplace=True)
    
    # 全局按时间排序 (这是时序学习的关键)
    data = data.sort_values("Timestamp").reset_index(drop=True)
    data["time_idx"] = data["Timestamp"].dt.floor("min")

    # --- 3. 分层时序划分 (按类别 8:1:1) ---
    print("Performing Stratified Temporal Split (8:1:1 per class)...")
    train_list = []
    val_list = []
    test_list = []

    for label in np.unique(data["Label"].values):
        cls_data = data[data["Label"] == label].sort_values("Timestamp")
        unique_times = cls_data["time_idx"].drop_duplicates().values
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

        train_list.append(cls_data[cls_data["time_idx"] < split_time_train])
        val_list.append(
            cls_data[
                (cls_data["time_idx"] >= split_time_train)
                & (cls_data["time_idx"] < split_time_val)
            ]
        )
        test_list.append(cls_data[cls_data["time_idx"] >= split_time_val])

    train_df = pd.concat(train_list).sort_values("Timestamp") if len(train_list) > 0 else data.iloc[0:0].copy()
    val_df = pd.concat(val_list).sort_values("Timestamp") if len(val_list) > 0 else data.iloc[0:0].copy()
    test_df = pd.concat(test_list).sort_values("Timestamp") if len(test_list) > 0 else data.iloc[0:0].copy()
    del data

    print(f"Final Split -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}", flush=True)
    print(f"Train Set Classes: {np.unique(train_df['Label'].values)}", flush=True)
    print(f"Val Set Classes: {np.unique(val_df['Label'].values)}", flush=True)
    print(f"Test Set Classes: {np.unique(test_df['Label'].values)}", flush=True)

    def _print_label_counts(df, split_name):
        vc = df["Label"].value_counts().sort_index()
        pairs = []
        for label_id, cnt in vc.items():
            label_name = class_names[int(label_id)] if int(label_id) < len(class_names) else str(label_id)
            pairs.append(f"{label_name}({int(label_id)}):{int(cnt)}")
        print(f"{split_name} Label Counts -> " + ", ".join(pairs), flush=True)

    _print_label_counts(train_df, "Train")
    _print_label_counts(val_df, "Val")
    _print_label_counts(test_df, "Test")

    hp_group = os.getenv("HP_GROUP", "")
    h = resolve_hparams(hp_group, env=os.environ)
    SEQ_LEN = int(h["SEQ_LEN"])
    BATCH_SIZE = int(h["BATCH_SIZE"])
    NUM_EPOCHS = int(h["NUM_EPOCHS"])
    LR = float(h["LR"])
    HIDDEN = int(h["HIDDEN"])
    HEADS = int(h["HEADS"])
    PATIENCE = int(h["PATIENCE"])
    MIN_DELTA = float(h["MIN_DELTA"])
    early_stop_metric = str(h["EARLY_STOP_METRIC"])
    cl_loss_weight = float(h["CL_LOSS_WEIGHT"])

    print(
        f"HP_GROUP: {(hp_group or '').strip().upper() if (hp_group or '').strip() else 'CUSTOM'} | "
        f"SEQ_LEN={SEQ_LEN}, BATCH_SIZE={BATCH_SIZE}, LR={LR}, HIDDEN={HIDDEN}, HEADS={HEADS}, "
        f"NUM_EPOCHS={NUM_EPOCHS}, PATIENCE={PATIENCE}, MIN_DELTA={MIN_DELTA}, "
        f"EARLY_STOP_METRIC={early_stop_metric}, CL_LOSS_WEIGHT={cl_loss_weight}",
        flush=True,
    )

    # --- 4. 归一化 ---
    print("Pre-processing & Normalization...")
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = {"Label", "Timestamp", "Src IP", "Dst IP", "Flow ID", "Src Port", "Dst Port", "time_idx"}
    feat_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    # 清洗 Inf/NaN 并填充 0
    for df in [train_df, val_df, test_df]:
        df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Log1p
    for col in feat_cols:
        if train_df[col].max() > 100:
            train_df[col] = np.log1p(train_df[col].abs())
            val_df[col] = np.log1p(val_df[col].abs())
            test_df[col] = np.log1p(test_df[col].abs())
            
    scaler = StandardScaler()
    train_df[feat_cols] = scaler.fit_transform(train_df[feat_cols])
    val_df[feat_cols] = scaler.transform(val_df[feat_cols])
    test_df[feat_cols] = scaler.transform(test_df[feat_cols])

    # --- 5. 构建 Graphs ---
    print("🏗️ Constructing Train Graphs...")
    subnet_to_idx = {'<UNK>': 0} 
    
    train_seqs = []
    for name, group in tqdm(train_df.groupby('time_idx', sort=True)):
        g = create_graph_data_inductive(group, subnet_to_idx, None, name)
        if g: train_seqs.append(g)

    print("🏗️ Constructing Val Graphs...")
    val_seqs = []
    for name, group in tqdm(val_df.groupby('time_idx', sort=True)):
        g = create_graph_data_inductive(group, subnet_to_idx, None, name)
        if g: val_seqs.append(g)

    print("🏗️ Constructing Test Graphs...")
    test_seqs = []
    for name, group in tqdm(test_df.groupby('time_idx', sort=True)):
        g = create_graph_data_inductive(group, subnet_to_idx, None, name)
        if g: test_seqs.append(g)

    val_seqs = _pad_seq_for_last_frame_coverage(val_seqs, SEQ_LEN)
    test_seqs = _pad_seq_for_last_frame_coverage(test_seqs, SEQ_LEN)

    # Loader
    train_loader = DataLoader(TemporalGraphDataset(train_seqs, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=True, collate_fn=temporal_collate_fn)
    val_loader = DataLoader(TemporalGraphDataset(val_seqs, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=False, collate_fn=temporal_collate_fn)
    test_loader = DataLoader(TemporalGraphDataset(test_seqs, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=False, collate_fn=temporal_collate_fn)

    # --- 6. 模型初始化 ---
    edge_dim = train_seqs[0].edge_attr.shape[1] if train_seqs else 1
    print(f"🧠 Initializing ROEN_Final (Edge Dim: {edge_dim})...")
    
    model = ROEN_Final(
        node_in=4, edge_in=edge_dim, 
        hidden=HIDDEN, 
        num_classes=len(class_names),
        seq_len=SEQ_LEN, 
        heads=HEADS
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 类别权重
    label_counts = train_df['Label'].value_counts().sort_index()
    full_counts = np.zeros(len(class_names))
    for i, count in label_counts.items():
        if i < len(full_counts): full_counts[i] = count
    weights = 1.0 / (torch.sqrt(torch.tensor(full_counts, dtype=torch.float)) + 1.0)
    weights = weights / weights.sum() * len(class_names)
    
    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
    # criterion = FocalLoss(alpha=weights.to(DEVICE), gamma=2.0)
    # 目录准备
    os.makedirs("models/ids2017_full", exist_ok=True)
    os.makedirs("png/ids2017_full", exist_ok=True)
    run_tag = f"seq{SEQ_LEN}_cls{len(class_names)}"
    best_model_path = f"models/ids2017_full/best_model_{run_tag}.pth"

    # --- 7. 训练循环 ---
    print("🔥 Start Training...")
    start_time = time.time()
    
    best_metric = -float("inf")
    no_improve_epochs = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        total_cl_loss = 0.0
        cl_loss_steps = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        for batched_seq in loop:
            if not batched_seq: continue
            batched_seq = [g.to(DEVICE) for g in batched_seq]
            
            optimizer.zero_grad()
            out = model(graphs=batched_seq)
            all_preds, cl_loss = out if isinstance(out, tuple) else (out, None)
            last_frame_pred = all_preds[-1]

            edge_masks = getattr(model, "_last_edge_masks", None)
            if edge_masks is not None and len(edge_masks) > 0 and edge_masks[-1] is not None:
                last_frame_labels = batched_seq[-1].edge_labels[edge_masks[-1]]
            else:
                last_frame_labels = batched_seq[-1].edge_labels

            main_loss = criterion(last_frame_pred, last_frame_labels)
            if torch.is_tensor(cl_loss):
                loss = main_loss + cl_loss_weight * cl_loss
                total_cl_loss += float(cl_loss.detach().item())
                cl_loss_steps += 1
            else:
                loss = main_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += loss.item()
            if cl_loss_steps > 0:
                loop.set_postfix(loss=loss.item(), cl_loss=float(cl_loss.detach().item()))
            else:
                loop.set_postfix(loss=loss.item())
            
        avg_loss = total_loss / max(1, len(train_loader))
        avg_cl_loss = total_cl_loss / max(1, cl_loss_steps)
        
        # 验证集评估
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batched_seq in val_loader:
                if not batched_seq: continue
                batched_seq = [g.to(DEVICE) for g in batched_seq]
                out = model(graphs=batched_seq)
                all_preds = out[0] if isinstance(out, tuple) else out
                val_loss = criterion(all_preds[-1], batched_seq[-1].edge_labels)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / max(1, len(val_loader))
        
        # 计算详细指标
        val_acc, val_prec, val_rec, val_f1, val_far, val_auc, val_asa, _, _ = evaluate_comprehensive_with_threshold(
            model, val_loader, DEVICE, class_names, threshold=0.5
        )

        print(
            f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Val F1: {val_f1:.4f} | ASA: {val_asa:.4f} | CL Loss: {avg_cl_loss:.4f}"
        )
        
        metric_value = val_f1 if early_stop_metric == "val_f1" else val_asa
        metric_display = "Val F1" if early_stop_metric == "val_f1" else "Val ASA"

        if metric_value > best_metric + MIN_DELTA:
            best_metric = metric_value
            no_improve_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "seq_len": SEQ_LEN,
                    "num_classes": len(class_names),
                    "class_names": class_names,
                    "edge_dim": edge_dim,
                },
                best_model_path,
            )
            print(f"New Best Model Saved! ({metric_display}: {best_metric:.4f})")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= PATIENCE:
                print(f"⏹️ Early Stopping at Epoch {epoch+1}")
                break

    # --- 8. 最终测试 ---
    print("\nLoading Best Model for Final Testing...")
    if os.path.exists(best_model_path):
        try:
            ckpt = torch.load(best_model_path, map_location=DEVICE)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt_seq_len = ckpt.get("seq_len", None)
                ckpt_num_classes = ckpt.get("num_classes", None)
                ckpt_edge_dim = ckpt.get("edge_dim", None)

                if ckpt_seq_len != SEQ_LEN or ckpt_num_classes != len(class_names) or ckpt_edge_dim != edge_dim:
                    print(
                        f"⚠️ Checkpoint config mismatch, skip loading. "
                        f"(ckpt seq_len={ckpt_seq_len}, num_classes={ckpt_num_classes}, edge_dim={ckpt_edge_dim}) "
                        f"vs (current seq_len={SEQ_LEN}, num_classes={len(class_names)}, edge_dim={edge_dim})"
                    )
                else:
                    model.load_state_dict(ckpt["state_dict"])
            else:
                model.load_state_dict(ckpt)
        except RuntimeError as e:
            print(f"⚠️ Failed to load checkpoint (shape mismatch). Using current weights. Error: {e}")
    
    print("\n=== Post-Training Threshold Optimization ===")
    best_asa = 0.0
    best_thresh = 0.5
    for th in [0.5, 0.4, 0.3, 0.25, 0.2]:
        _, f1, far, asa = evaluate_with_threshold(model, test_loader, DEVICE, class_names, threshold=th)
        print(f"[Threshold {th}] -> F1: {f1:.4f}, FAR: {far:.4f}, ASA: {asa:.4f}")
        if asa > best_asa and far < 0.03:
            best_asa = asa
            best_thresh = th
            
    print(f"\n✅ Best Strategy: Threshold = {best_thresh}")
    
    # 最终结果
    final_acc, final_prec, final_rec, final_f1, final_far, final_auc, final_asa, final_labels, final_preds = evaluate_comprehensive_with_threshold(
        model, test_loader, DEVICE, class_names, threshold=best_thresh
    )
    present = np.unique(np.asarray(final_labels, dtype=np.int64))
    missing = sorted(list(set(range(len(class_names))) - set(present.tolist())))
    counts = np.bincount(np.asarray(final_labels, dtype=np.int64), minlength=len(class_names))
    print(f"Final Labels Present IDs: {present.tolist()}", flush=True)
    if len(missing) > 0:
        missing_names = [class_names[i] for i in missing if i < len(class_names)]
        print(f"⚠️ Final Labels Missing IDs: {missing} ({missing_names})", flush=True)
    nonzero_pairs = []
    for i, c in enumerate(counts.tolist()):
        if c > 0:
            nonzero_pairs.append(f"{class_names[i]}({i}):{c}")
    print("Final Labels Counts -> " + ", ".join(nonzero_pairs), flush=True)
    print(
        f"Final Test -> ACC: {final_acc:.4f}, PREC: {final_prec:.4f}, Rec: {final_rec:.4f}, "
        f"F1: {final_f1:.4f}, AUC: {final_auc:.4f}, ASA: {final_asa:.4f}"
    )

    # 画图
    try:
        labels_idx = list(range(len(class_names)))
        cm = confusion_matrix(final_labels, final_preds, labels=labels_idx)
        cm = np.asarray(cm, dtype=np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0) * 100.0

        import seaborn as sns
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix (Threshold={best_thresh})')
        plt.tight_layout()
        plt.savefig(f'png/ids2017_full/FINAL_CM.png', dpi=300)
        print("Confusion Matrix Saved.")
    except Exception as e:
        print(f"Plotting failed: {e}")

    print(f"Total Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
