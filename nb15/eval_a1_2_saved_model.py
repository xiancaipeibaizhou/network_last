import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch_geometric.loader import DataLoader

from a1_2 import UNK_SUBNET_ID, TemporalGraphDataset, _subnet_key, create_graph_data_inductive, temporal_collate_fn
from model_Final import ROEN_Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _get_normal_indices(class_names):
    keywords = ("non", "non-tor", "nonvpn", "normal", "benign")
    normal_indices = []
    for idx, name in enumerate(class_names):
        low = str(name).lower()
        if any(k in low for k in keywords):
            normal_indices.append(idx)
    if len(normal_indices) == 0 and len(class_names) > 0:
        normal_indices = [0]
    return normal_indices


def _auc_ovr_macro(y_true, y_probs, present_labels):
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs)
    if y_probs.ndim != 2 or y_probs.shape[0] != y_true.shape[0]:
        return float("nan")

    aucs = []
    for c in np.asarray(present_labels).astype(int):
        if c < 0 or c >= y_probs.shape[1]:
            continue
        y_bin = (y_true == c).astype(int)
        if np.unique(y_bin).size < 2:
            continue
        try:
            aucs.append(float(roc_auc_score(y_bin, y_probs[:, c])))
        except Exception:
            continue
    if len(aucs) == 0:
        return float("nan")
    return float(np.mean(aucs))


def _preds_with_threshold(probs, class_names, threshold):
    probs = np.asarray(probs)
    if probs.ndim != 2 or probs.shape[0] == 0:
        return np.zeros((probs.shape[0],), dtype=int)

    normal_indices = _get_normal_indices(class_names)
    attack_indices = [i for i in range(len(class_names)) if i not in set(normal_indices)]

    preds = probs.argmax(axis=1).astype(int)
    if len(attack_indices) == 0:
        return preds

    attack_prob_sum = probs[:, attack_indices].sum(axis=1)
    mask = attack_prob_sum > float(threshold)
    if mask.any():
        sub_probs = probs[mask][:, attack_indices]
        sub_argmax = sub_probs.argmax(axis=1).astype(int)
        preds[mask] = np.asarray(attack_indices, dtype=int)[sub_argmax]
    return preds


def save_confusion_matrix(y_true, y_pred, class_names, save_path, normalize=True):
    labels_idx = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)
    cm = np.asarray(cm, dtype=np.float64)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_to_plot = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0) * 100.0
        fmt = ".1f"
        title_suffix = " (Row %)"
    else:
        cm_to_plot = cm
        fmt = ".0f"
        title_suffix = " (Count)"

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    try:
        import seaborn as sns

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_to_plot,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
    except Exception:
        plt.figure(figsize=(10, 8))
        plt.imshow(cm_to_plot, interpolation="nearest", cmap="Blues")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha="right")
        plt.yticks(tick_marks, class_names)
        for i in range(cm_to_plot.shape[0]):
            for j in range(cm_to_plot.shape[1]):
                plt.text(j, i, format(cm_to_plot[i, j], fmt), ha="center", va="center", fontsize=8)

    plt.title(f"Confusion Matrix{title_suffix}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


@torch.no_grad()
def evaluate_comprehensive_compatible(model, dataloader, device, class_names, threshold=None):
    model.eval()
    all_labels = []
    all_probs = []

    for batched_seq in dataloader:
        batched_seq = [g.to(device) for g in batched_seq]
        out = model(batched_seq)
        preds_seq = out[0] if isinstance(out, tuple) else out
        logits = preds_seq[-1]

        probs = torch.softmax(logits, dim=1)
        all_labels.extend(batched_seq[-1].edge_labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    if len(all_labels) == 0:
        metrics = {
            "acc": 0.0,
            "prec": 0.0,
            "rec": 0.0,
            "f1": 0.0,
            "far": 0.0,
            "auc": float("nan"),
            "asa": 0.0,
        }
        return metrics, np.asarray([], dtype=int), np.asarray([], dtype=int)

    y_true = np.asarray(all_labels).astype(int)
    y_probs = np.asarray(all_probs)
    if threshold is None:
        y_pred = y_probs.argmax(axis=1).astype(int)
    else:
        y_pred = _preds_with_threshold(y_probs, class_names, threshold)

    acc = float((y_pred == y_true).mean())
    prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    normal_indices = _get_normal_indices(class_names)
    is_true_normal = np.isin(y_true, normal_indices)
    is_pred_normal = np.isin(y_pred, normal_indices)
    fp = np.logical_and(is_true_normal, ~is_pred_normal).sum()
    tn = np.logical_and(is_true_normal, is_pred_normal).sum()
    far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    is_true_attack = ~is_true_normal
    asa = float((y_pred[is_true_attack] == y_true[is_true_attack]).mean()) if is_true_attack.any() else 0.0

    try:
        present_labels = np.unique(y_true).astype(int)
        if y_probs.ndim != 2 or y_probs.shape[0] != y_true.shape[0] or present_labels.size < 2:
            auc = float("nan")
        elif y_probs.shape[1] == 2 or present_labels.size == 2:
            present_labels = np.sort(present_labels)
            pos_label = int(present_labels[-1])
            y_true_bin = (y_true == pos_label).astype(int)
            auc = float(roc_auc_score(y_true_bin, y_probs[:, pos_label]))
        else:
            auc = _auc_ovr_macro(y_true, y_probs, present_labels)
    except Exception:
        auc = float("nan")

    metrics = {
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "far": far,
        "auc": auc,
        "asa": asa,
    }
    return metrics, y_true, y_pred


def build_test_loader(data_path, seq_len, batch_size):
    data = pd.read_csv(data_path)

    data["Label"] = data["Label"].astype(str).str.strip().replace("", np.nan)
    data.dropna(subset=["Label", "Timestamp"], inplace=True)

    label_encoder = LabelEncoder()
    data["Label"] = label_encoder.fit_transform(data["Label"])
    class_names = list(label_encoder.classes_)

    data["Timestamp"] = pd.to_datetime(data["Timestamp"], dayfirst=True, errors="coerce")
    data.dropna(subset=["Timestamp"], inplace=True)
    data = data.sort_values("Timestamp")
    data["time_idx"] = data["Timestamp"].dt.floor("min")

    unique_times = data["time_idx"].drop_duplicates().values
    total_len = len(unique_times)
    train_idx = int(total_len * 0.8)
    val_idx = int(total_len * 0.9)
    train_idx = max(1, min(total_len - 1, train_idx))
    val_idx = max(train_idx + 1, min(total_len - 1, val_idx))

    split_time_train = unique_times[train_idx]
    split_time_val = unique_times[val_idx]

    train_df = data[data["time_idx"] < split_time_train].copy()
    test_df = data[data["time_idx"] >= split_time_val].copy()

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ["Label", "Timestamp", "Src IP", "Dst IP", "Flow ID", "Src Port", "Dst Port"]
    feat_cols = [c for c in numeric_cols if c not in exclude]

    train_df[feat_cols] = train_df[feat_cols].fillna(0)
    test_df[feat_cols] = test_df[feat_cols].fillna(0)

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

    test_grouped = test_df.groupby("time_idx", sort=True)
    test_seqs = []
    for _, group in test_grouped:
        g = create_graph_data_inductive(group, subnet_to_idx, None, None)
        if g is not None:
            test_seqs.append(g)

    test_dataset = TemporalGraphDataset(test_seqs, seq_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=temporal_collate_fn)

    edge_dim = 1
    if len(test_seqs) > 0:
        edge_dim = int(test_seqs[0].edge_attr.shape[1])

    return test_loader, class_names, edge_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/CIC-NUSW-NB15/CICFlowMeter_out.csv")
    parser.add_argument("--ckpt", default="models/nb15/best_model_enhanced.pth")
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--thresh", type=float, default=0.4)
    parser.add_argument("--no-cm", action="store_true")
    parser.add_argument("--cm-path", default=None)
    parser.add_argument("--cm-no-normalize", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    test_loader, class_names, edge_dim = build_test_loader(args.data, args.seq_len, args.batch_size)
    model = ROEN_Final(
        node_in=4,
        edge_in=edge_dim,
        hidden=args.hidden,
        num_classes=len(class_names),
        seq_len=args.seq_len,
        heads=args.heads,
    ).to(device)

    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)

    print(f"Checkpoint: {args.ckpt}")
    metrics, y_true, y_pred = evaluate_comprehensive_compatible(model, test_loader, device, class_names, threshold=args.thresh)
    print(
        f"Thresh {args.thresh}:"
        "Test -> "
        f"ACC: {metrics['acc']:.4f}, "
        f"Prec: {metrics['prec']:.4f}, "
        f"Rec: {metrics['rec']:.4f}, "
        f"F1: {metrics['f1']:.4f}, "
        f"FAR: {metrics['far']:.4f}, "
        f"AUC: {metrics['auc']:.4f}, "
        f"ASA: {metrics['asa']:.4f}"
    )

    if not args.no_cm and y_true.size > 0:
        if args.cm_path is None:
            th_str = str(args.thresh).replace(".", "p")
            cm_path = os.path.join("png", "nb15", f"CM_Thresh{th_str}.png")
        else:
            cm_path = args.cm_path

        save_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            save_path=cm_path,
            normalize=not args.cm_no_normalize,
        )
        print(f"Saved CM to {cm_path}")


if __name__ == "__main__":
    main()
