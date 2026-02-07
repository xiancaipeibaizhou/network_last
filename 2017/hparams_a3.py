import os


GROUPS = {
    "A1": {"SEQ_LEN": 32, "BATCH_SIZE": 8},
    "A2": {"SEQ_LEN": 64, "BATCH_SIZE": 4},
    "B1": {"HIDDEN": 64, "HEADS": 8},
    "B2": {"HIDDEN": 128, "HEADS": 4},
    "C1": {"CL_LOSS_WEIGHT": 0.5},
    "C2": {"CL_LOSS_WEIGHT": 1.0, "LR": 0.0005},
    "D1": {"SEQ_LEN": 32, "HIDDEN": 64, "CL_LOSS_WEIGHT": 0.2},
}


def resolve_hparams(group, env=None):
    if env is None:
        env = os.environ

    h = {
        "SEQ_LEN": int(env.get("SEQ_LEN", "10")),
        "BATCH_SIZE": int(env.get("BATCH_SIZE", "16")),
        "NUM_EPOCHS": int(env.get("NUM_EPOCHS", "150")),
        "LR": float(env.get("LR", "0.001")),
        "HIDDEN": int(env.get("HIDDEN", "32")),
        "HEADS": int(env.get("HEADS", "8")),
        "PATIENCE": int(env.get("PATIENCE", "10")),
        "MIN_DELTA": float(env.get("MIN_DELTA", "0.0")),
        "EARLY_STOP_METRIC": str(env.get("EARLY_STOP_METRIC", "val_f1")).strip().lower(),
        "CL_LOSS_WEIGHT": float(env.get("CL_LOSS_WEIGHT", "0.01")),
    }

    group = (group or "").strip().upper()
    if group in GROUPS:
        for k, v in GROUPS[group].items():
            if k not in env:
                h[k] = v

    if h["EARLY_STOP_METRIC"] in {"f1", "valf1"}:
        h["EARLY_STOP_METRIC"] = "val_f1"
    elif h["EARLY_STOP_METRIC"] in {"asa", "valasa"}:
        h["EARLY_STOP_METRIC"] = "val_asa"
    else:
        h["EARLY_STOP_METRIC"] = "val_f1"

    return h

