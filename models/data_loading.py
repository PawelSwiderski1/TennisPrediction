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


def prepare_single_row(
    row,
    p1_info_cols,
    p2_info_cols,
    env_cols,
    scalers,
    player_vocab,
    device,
):
    """
    Prepare a single row for model inference using pre-fitted scalers.

    Args:
        row: A single row from the DataFrame (pd.Series or dict-like)
        p1_info_cols: List of player 1 info column names
        p2_info_cols: List of player 2 info column names
        env_cols: List of environment column names
        scalers: Dict containing fitted scalers with keys:
            - 'elo_scaler': StandardScaler for elo columns
            - 'info_residual_scaler': MinMaxScaler for other info columns
            - 'env_scaler': MinMaxScaler for environment columns
            - 'form_params': Dict with 'surplus_c' and 'days_cap'
        player_vocab: Dict mapping player_id -> index
        device: torch device

    Returns:
        Tuple of tensors: (p1_info, p1_form, p2_info, p2_form, env)
    """
    elo_scaler = scalers["elo_scaler"]
    residual_scaler = scalers["info_residual_scaler"]
    env_scaler = scalers["env_scaler"]
    form_params = scalers["form_params"]
    days_cap = form_params["days_cap"]

    # Player indices
    p1_id = int(row["player1_id"])
    p2_id = int(row["player2_id"])
    p1_idx = player_vocab.get(p1_id, 0)
    p2_idx = player_vocab.get(p2_id, 0)

    # Transform info features for a single row
    def transform_info_single(row_data, cols, elo_sc, residual_sc):
        X = pd.DataFrame([{c: float(row_data[c]) if pd.notna(row_data[c]) else 0.0 for c in cols}])
        X = X.astype(np.float32)

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
            X[residual_cols] = residual_sc.transform(vals)

        if elo_cols:
            vals = safe(X[elo_cols].values)
            X[elo_cols] = elo_sc.transform(vals)

        return X.values.astype(np.float32).flatten()

    p1_info = transform_info_single(row, p1_info_cols, elo_scaler, residual_scaler)
    p2_info = transform_info_single(row, p2_info_cols, elo_scaler, residual_scaler)

    # Environment features
    env = np.array([[float(row[c]) if pd.notna(row[c]) else 0.0 for c in env_cols]], dtype=np.float32)
    env = env_scaler.transform(env).flatten()

    # Form features - use same stacking as extract_form: stack columns then transpose
    p1_form_cols = [f"player1_{s}" for s in FORM_SUFFIXES]
    p2_form_cols = [f"player2_{s}" for s in FORM_SUFFIXES]

    # Stack arrays along axis 0 (one array per feature), then transpose to (T, D)
    p1_form_arrays = [to_array(row[c]) for c in p1_form_cols]  # D arrays of shape (T,)
    p1_form = np.stack(p1_form_arrays, axis=0).T.astype(np.float32)  # (T, D)

    p2_form_arrays = [to_array(row[c]) for c in p2_form_cols]
    p2_form = np.stack(p2_form_arrays, axis=0).T.astype(np.float32)  # (T, D)

    # Apply form scaling using fitted elo_scaler stats (same as process_form)
    if hasattr(elo_scaler, 'mean_') and len(elo_scaler.mean_) > 1:
        mu = elo_scaler.mean_[1]  # standard elo dimension
        sd = max(np.sqrt(elo_scaler.var_[1]), 1e-6)
        p1_form[:, IDX_OPP_ELO] = (safe(p1_form[:, IDX_OPP_ELO]) - mu) / sd
        p2_form[:, IDX_OPP_ELO] = (safe(p2_form[:, IDX_OPP_ELO]) - mu) / sd

    # surplus and days scaling (use constants, same as process_form)
    p1_form[:, IDX_SURPLUS] = np.clip(p1_form[:, IDX_SURPLUS] / SURPLUS_SCALE, -1, 1)
    p2_form[:, IDX_SURPLUS] = np.clip(p2_form[:, IDX_SURPLUS] / SURPLUS_SCALE, -1, 1)

    p1_days = np.maximum(p1_form[:, IDX_DAYS], 0.0)
    p2_days = np.maximum(p2_form[:, IDX_DAYS], 0.0)
    p1_form[:, IDX_DAYS] = np.clip(np.log1p(p1_days) / days_cap, 0, 1)
    p2_form[:, IDX_DAYS] = np.clip(np.log1p(p2_days) / days_cap, 0, 1)

    # Convert to tensors with batch dimension
    p1_info_t = torch.tensor(p1_info, dtype=torch.float32, device=device).unsqueeze(0)
    p2_info_t = torch.tensor(p2_info, dtype=torch.float32, device=device).unsqueeze(0)
    p1_form_t = torch.tensor(p1_form, dtype=torch.float32, device=device).unsqueeze(0)
    p2_form_t = torch.tensor(p2_form, dtype=torch.float32, device=device).unsqueeze(0)
    env_t = torch.tensor(env, dtype=torch.float32, device=device).unsqueeze(0)

    return p1_info_t, p1_form_t, p2_info_t, p2_form_t, env_t


def compare_single_vs_batch(row_idx, test_df, p1_info_cols, p2_info_cols, env_cols, scalers, player_vocab, device):
    """
    Debug function to compare prepare_single_row output vs batch processing.
    Returns dict with differences for each tensor.
    """
    # Get single row result
    row = test_df.iloc[row_idx]
    single_p1_info, single_p1_form, single_p2_info, single_p2_form, single_env = prepare_single_row(
        row, p1_info_cols, p2_info_cols, env_cols, scalers, player_vocab, device
    )

    # Get batch result for comparison
    cols = {"p1_info": p1_info_cols, "p2_info": p2_info_cols, "env": env_cols}
    batch_scalers = {
        "elo": scalers["elo_scaler"],
        "residual": scalers["info_residual_scaler"],
        "env": scalers["env_scaler"],
        "days_cap": scalers["form_params"]["days_cap"],
    }

    # Process just this one row as a dataframe
    single_row_df = test_df.iloc[[row_idx]]
    batch_data = process_fold(single_row_df, cols, batch_scalers, player_vocab, fit=False)

    batch_p1_info = torch.tensor(batch_data["p1_info"], dtype=torch.float32, device=device)
    batch_p2_info = torch.tensor(batch_data["p2_info"], dtype=torch.float32, device=device)
    batch_p1_form = torch.tensor(batch_data["p1_form"], dtype=torch.float32, device=device)
    batch_p2_form = torch.tensor(batch_data["p2_form"], dtype=torch.float32, device=device)
    batch_env = torch.tensor(batch_data["env"], dtype=torch.float32, device=device)

    results = {
        "p1_info_match": torch.allclose(single_p1_info.squeeze(), batch_p1_info.squeeze(), atol=1e-5),
        "p2_info_match": torch.allclose(single_p2_info.squeeze(), batch_p2_info.squeeze(), atol=1e-5),
        "p1_form_match": torch.allclose(single_p1_form.squeeze(), batch_p1_form.squeeze(), atol=1e-5),
        "p2_form_match": torch.allclose(single_p2_form.squeeze(), batch_p2_form.squeeze(), atol=1e-5),
        "env_match": torch.allclose(single_env.squeeze(), batch_env.squeeze(), atol=1e-5),
        "p1_info_diff": (single_p1_info.squeeze() - batch_p1_info.squeeze()).abs().max().item(),
        "p2_info_diff": (single_p2_info.squeeze() - batch_p2_info.squeeze()).abs().max().item(),
        "p1_form_diff": (single_p1_form.squeeze() - batch_p1_form.squeeze()).abs().max().item(),
        "p2_form_diff": (single_p2_form.squeeze() - batch_p2_form.squeeze()).abs().max().item(),
        "env_diff": (single_env.squeeze() - batch_env.squeeze()).abs().max().item(),
    }

    return results


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
