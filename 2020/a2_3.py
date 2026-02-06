import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from analys import evaluate_comprehensive
from network_new import ROEN_Fast_Transformer
from a2_1 import UNK_SUBNET_ID, TemporalGraphDataset, _subnet_key, create_graph_data_inductive, temporal_collate_fn


def _build_graphs(df, subnet_to_idx, title):
    grouped = df.groupby("time_idx", sort=True)
    seqs = []
    for name, group in tqdm(grouped, desc=title):
        g = create_graph_data_inductive(group, subnet_to_idx, None, name)
        if g:
            seqs.append(g)
    return seqs


def _collect_last_frame_preds(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batched_seq in dataloader:
            if not batched_seq:
                continue
            batched_seq = [g.to(device) for g in batched_seq]
            try:
                preds_seq = model(graphs=batched_seq, seq_len=len(batched_seq))
            except TypeError:
                preds_seq = model(graphs=batched_seq)
            logits = preds_seq[-1]
            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            all_labels.extend(batched_seq[-1].edge_labels.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())

    return np.asarray(all_labels), np.asarray(all_preds)


def main():
    SEQ_LEN = int(os.getenv("SEQ_LEN", "10"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "100"))
    LR = float(os.getenv("LR", "0.001"))
    CSV_PATH = os.getenv("CSV_PATH", "data/CIC-Darknet2020/Darknet.csv")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {DEVICE}")

    data = pd.read_csv(CSV_PATH)
    data.drop(columns=["Label.1"], inplace=True, errors="ignore")
    data = data.dropna(subset=["Label", "Timestamp"]).copy()

    label_encoder = LabelEncoder()
    data["Label"] = label_encoder.fit_transform(data["Label"].astype(str))
    class_names = list(label_encoder.classes_)
    print(f"Classes: {class_names}")

    data["Timestamp"] = pd.to_datetime(data["Timestamp"], dayfirst=True, errors="coerce")
    data = data.dropna(subset=["Timestamp"])
    data = data.sort_values("Timestamp")
    data["time_idx"] = data["Timestamp"].dt.floor("min")

    train_list = []
    test_list = []
    for label in data["Label"].unique():
        cls_data = data[data["Label"] == label].sort_values("Timestamp")
        unique_times = cls_data["time_idx"].drop_duplicates().values
        total_len = len(unique_times)
        if total_len < 2:
            train_list.append(cls_data)
            continue

        split_idx = int(total_len * 0.8)
        split_idx = max(1, min(split_idx, total_len - 1))
        split_time = unique_times[split_idx]

        train_list.append(cls_data[cls_data["time_idx"] < split_time])
        test_list.append(cls_data[cls_data["time_idx"] >= split_time])

    train_df = pd.concat(train_list).sort_values("Timestamp") if len(train_list) > 0 else data.iloc[0:0].copy()
    test_df = pd.concat(test_list).sort_values("Timestamp") if len(test_list) > 0 else data.iloc[0:0].copy()
    print(f"Final Split -> Train: {len(train_df)}, Test: {len(test_df)}", flush=True)

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ["Label", "Timestamp", "Src IP", "Dst IP", "Flow ID", "Src Port", "Dst Port", "time_idx"]
    feat_cols = [c for c in numeric_cols if c not in exclude_cols]

    train_df[feat_cols] = train_df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    test_df[feat_cols] = test_df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    for col in feat_cols:
        if train_df[col].max() > 100:
            train_df[col] = np.log1p(train_df[col].abs())
            test_df[col] = np.log1p(test_df[col].abs())

    scaler = StandardScaler()
    train_df[feat_cols] = scaler.fit_transform(train_df[feat_cols])
    test_df[feat_cols] = scaler.transform(test_df[feat_cols])

    train_df["Src IP"] = train_df["Src IP"].astype(str).str.strip()
    train_df["Dst IP"] = train_df["Dst IP"].astype(str).str.strip()
    train_ips = pd.concat([train_df["Src IP"], train_df["Dst IP"]]).unique()

    subnet_to_idx = {"<UNK>": UNK_SUBNET_ID}
    for ip in train_ips:
        key = _subnet_key(ip)
        if key not in subnet_to_idx:
            subnet_to_idx[key] = len(subnet_to_idx)
    num_subnets = len(subnet_to_idx)
    print(f"Train Subnets: {num_subnets}")

    train_seqs = _build_graphs(train_df, subnet_to_idx, "Constructing Train Graphs")
    test_seqs = _build_graphs(test_df, subnet_to_idx, "Constructing Test Graphs")
    print(f"Total Train Graphs: {len(train_seqs)}, Test Graphs: {len(test_seqs)}")

    train_dataset = TemporalGraphDataset(train_seqs, seq_len=SEQ_LEN)
    test_dataset = TemporalGraphDataset(test_seqs, seq_len=SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=temporal_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=temporal_collate_fn)

    reference_graphs = train_seqs if len(train_seqs) > 0 else test_seqs
    edge_dim = reference_graphs[0].edge_attr.shape[1] if len(reference_graphs) > 0 else 1

    model = ROEN_Fast_Transformer(
        node_in=4,
        edge_in=edge_dim,
        hidden=128,
        num_classes=len(class_names),
        num_subnets=num_subnets,
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    label_counts = train_df["Label"].value_counts().sort_index()
    class_counts_tensor = torch.tensor(label_counts.values, dtype=torch.float).to(DEVICE)
    weights = 1.0 / (torch.sqrt(class_counts_tensor) + 1.0)
    weights = weights / weights.sum() * len(class_names)
    criterion = nn.CrossEntropyLoss(weight=weights)

    print(f"Start Training on {DEVICE}...")
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        for batched_seq in loop:
            if not batched_seq:
                continue
            batched_seq = [g.to(DEVICE) for g in batched_seq]

            optimizer.zero_grad()
            preds_seq = model(graphs=batched_seq, seq_len=len(batched_seq))
            loss = criterion(preds_seq[-1], batched_seq[-1].edge_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            loss_val = float(loss.detach().cpu().item())
            total_loss += loss_val
            loop.set_postfix(loss=loss_val)

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f}")

    os.makedirs("models/2020", exist_ok=True)
    model_save_path = "models/2020/a2_3_roen_fast_transformer_final.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    acc, prec, rec, f1, far, auc, asa = evaluate_comprehensive(model, test_loader, DEVICE, class_names)
    print(
        f"Final Test -> ACC: {acc:.4f}, PREC: {prec:.4f}, F1: {f1:.4f}, "
        f"Rec: {rec:.4f}, FAR: {far:.4f}, AUC: {auc:.4f}, ASA: {asa:.4f}"
    )

    final_labels, final_preds = _collect_last_frame_preds(model, test_loader, DEVICE)
    labels_idx = list(range(len(class_names)))
    cm = confusion_matrix(final_labels, final_preds, labels=labels_idx)
    cm = np.asarray(cm, dtype=np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0) * 100.0

    os.makedirs("png/2020", exist_ok=True)
    save_path = "png/2020/FINAL_TEST_CM_a2_3.png"
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_pct, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (%)")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm_pct.max() / 2.0 if cm_pct.size > 0 else 0.0
    for i in range(cm_pct.shape[0]):
        for j in range(cm_pct.shape[1]):
            plt.text(j, i, f"{cm_pct[i, j]:.1f}", ha="center", va="center", color="white" if cm_pct[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Final Confusion Matrix saved to {save_path}")
    print(f"Total Time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
