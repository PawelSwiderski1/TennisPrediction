import ast
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Form feature column order
FORM_SUFFIXES = [
    "last_opponent_elo", "last_base_perfs", "last_set_margin_norm",
    "last_game_margin_norm", "last_margin_surplus", "last_best_of_3",
    "last_same_surface", "last_same_tournament", "last_days_since",
]
IDX_OPP_ELO = 0
IDX_SURPLUS = 4
IDX_DAYS = 8

SURPLUS_SCALE = 2.3
DAYS_QUANTILE = 0.99


def to_array(x):
    if isinstance(x, str):
        return np.array(ast.literal_eval(x), dtype=np.float32)
    return np.array(x, dtype=np.float32)


def safe(arr):
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def build_player_vocab(df, id_cols=("player1_id", "player2_id")):
    ids = pd.concat([df[c].astype(int) for c in id_cols if c in df.columns])
    unique = pd.unique(ids)
    return {pid: i + 1 for i, pid in enumerate(sorted(unique))}


def is_elo_col(c):
    c = c.lower()
    return c.endswith("_elo") or "surface_elo" in c


def is_passthrough(c):
    patterns = ("right_handed", "entry_", "win_pct", "_history", "is_seeded")
    return any(p in c.lower() for p in patterns)


def transform_info(df, cols, elo_scaler, residual_scaler, fit=False):
    X = df[cols].copy().astype(np.float32)
    residual_cols, elo_cols = [], []

    for c in cols:
        if is_passthrough(c):
            continue
        if is_elo_col(c):
            elo_cols.append(c)
        else:
            residual_cols.append(c)

    if residual_cols:
        vals = safe(X[residual_cols].values)
        if fit:
            residual_scaler.fit(vals)
        X[residual_cols] = residual_scaler.transform(vals)

    if elo_cols:
        vals = safe(X[elo_cols].values)
        if fit:
            elo_scaler.fit(vals)
        X[elo_cols] = elo_scaler.transform(vals)

    return X.values.astype(np.float32)


def extract_form(df, prefix):
    cols = [f"{prefix}_{s}" for s in FORM_SUFFIXES]
    stacked = np.stack([np.stack([to_array(v) for v in df[c]]) for c in cols], axis=0)
    return np.transpose(stacked, (1, 2, 0))  # [N, T, D]


def process_form(df, elo_scaler, days_cap=None, fit=False):
    p1 = extract_form(df, "player1")
    p2 = extract_form(df, "player2")

    N, T, D = p1.shape
    p1_flat = p1.reshape(-1, D)
    p2_flat = p2.reshape(-1, D)

    # scale opponent elo using fitted scaler stats
    if hasattr(elo_scaler, 'mean_') and len(elo_scaler.mean_) > 1:
        mu = elo_scaler.mean_[1]
        sd = max(np.sqrt(elo_scaler.var_[1]), 1e-6)
        p1_flat[:, IDX_OPP_ELO] = (safe(p1_flat[:, IDX_OPP_ELO]) - mu) / sd
        p2_flat[:, IDX_OPP_ELO] = (safe(p2_flat[:, IDX_OPP_ELO]) - mu) / sd

    # fit days cap on train
    if fit:
        all_days = np.concatenate([p1_flat[:, IDX_DAYS], p2_flat[:, IDX_DAYS]])
        all_days = np.maximum(all_days, 0.0)
        days_cap = max(np.quantile(np.log1p(all_days), DAYS_QUANTILE), 1e-6)

    # transform
    for flat in [p1_flat, p2_flat]:
        flat[:, IDX_SURPLUS] = np.clip(flat[:, IDX_SURPLUS] / SURPLUS_SCALE, -1, 1)
        days = np.maximum(flat[:, IDX_DAYS], 0.0)
        flat[:, IDX_DAYS] = np.clip(np.log1p(days) / days_cap, 0, 1)

    return p1_flat.reshape(N, T, D), p2_flat.reshape(N, T, D), days_cap


def make_dataset(device, *arrays):
    tensors = []
    for arr in arrays:
        arr = np.array(arr)
        if arr.dtype == object:
            arr = np.stack(arr).astype(np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        dtype = torch.int32 if np.issubdtype(arr.dtype, np.integer) else torch.float32
        tensors.append(torch.tensor(arr, dtype=dtype, device=device))
    return TensorDataset(*tensors)


def process_fold(df, cols, scalers, player_vocab, fit=False):
    p1_idx = df["player1_id"].map(player_vocab).fillna(0).astype("int32").values
    p2_idx = df["player2_id"].map(player_vocab).fillna(0).astype("int32").values

    p1_info = transform_info(df, cols["p1_info"], scalers["elo"], scalers["residual"], fit=fit)
    p2_info = transform_info(df, cols["p2_info"], scalers["elo"], scalers["residual"], fit=False)

    env = df[cols["env"]].values if cols["env"] else np.zeros((len(df), 1))
    if fit:
        env = scalers["env"].fit_transform(env)
    else:
        env = scalers["env"].transform(env)

    p1_form, p2_form, days_cap = process_form(df, scalers["elo"], scalers.get("days_cap"), fit=fit)
    if fit:
        scalers["days_cap"] = days_cap

    y = df["target"].values

    return {
        "p1_idx": p1_idx, "p2_idx": p2_idx,
        "p1_info": p1_info, "p2_info": p2_info,
        "p1_form": p1_form, "p2_form": p2_form,
        "env": env, "y": y,
    }


def prepare_dataloaders(
    train_df, val_df,
    p1_info_cols, p2_info_cols,
    env_cols,
    batch_size, player_vocab, device,
    test_df=None,
):
    cols = {
        "p1_info": p1_info_cols, "p2_info": p2_info_cols,
        "env": env_cols,
    }

    scalers = {
        "elo": StandardScaler(),
        "residual": MinMaxScaler(),
        "env": MinMaxScaler(),
    }

    # train
    train_data = process_fold(train_df, cols, scalers, player_vocab, fit=True)
    train_arrays = [
        train_data["p1_idx"], train_data["p2_idx"],
        train_data["p1_info"], train_data["p1_form"],
        train_data["p2_info"], train_data["p2_form"],
        train_data["env"], train_data["y"],
    ]
    train_loader = DataLoader(make_dataset(device, *train_arrays), batch_size=batch_size, shuffle=True, drop_last=True)

    # val
    val_data = process_fold(val_df, cols, scalers, player_vocab, fit=False)
    val_arrays = [
        val_data["p1_idx"], val_data["p2_idx"],
        val_data["p1_info"], val_data["p1_form"],
        val_data["p2_info"], val_data["p2_form"],
        val_data["env"], val_data["y"],
    ]
    val_loader = DataLoader(make_dataset(device, *val_arrays), batch_size=batch_size, shuffle=False)

    scaler_dict = {
        "elo_scaler": scalers["elo"],
        "info_residual_scaler": scalers["residual"],
        "env_scaler": scalers["env"],
        "form_params": {"surplus_c": SURPLUS_SCALE, "days_cap": scalers.get("days_cap", 1.0)},
    }

    if test_df is None:
        return train_loader, val_loader, scaler_dict

    # test
    test_data = process_fold(test_df, cols, scalers, player_vocab, fit=False)
    test_arrays = [
        test_data["p1_idx"], test_data["p2_idx"],
        test_data["p1_info"], test_data["p1_form"],
        test_data["p2_info"], test_data["p2_form"],
        test_data["env"], test_data["y"],
    ]
    for c in ["player1_rank", "player2_rank", "player1_bet_odds", "player2_bet_odds"]:
        if c in test_df.columns:
            test_arrays.append(test_df[c].values.reshape(-1, 1))

    test_arrays.append(np.arange(len(test_df))) # match_id_key

    test_loader = DataLoader(make_dataset(device, *test_arrays), batch_size=batch_size, shuffle=False)
    match_ids = test_df["match_id"].values

    return train_loader, val_loader, test_loader, match_ids, scaler_dict
