"""
Models:
  - Logistic Regression (sklearn)
  - Random Forest (sklearn)
  - Gradient Boosting (sklearn)
  - MLP (PyTorch)
  - Tiny Transformer (PyTorch) on N-round sequences per match
  - TD(0) Value Net (PyTorch) — RL-flavored baseline

Inputs:
  A CSV with columns:
    - match_id (str)
    - round_num (int)
    - side (str: "T"/"CT")
    - y (int: 0/1 team won the round)
    - ... arbitrary numeric feature columns (e.g., dmg_T, dmg_CT, first_kill_T, etc.)

Usage:
  python model_zoo.py --features clean_dataset/features_round_team.csv --seq_len 5 --epochs 15

Outputs:
  - models/*.joblib (sklearn) and *.pt (torch)
  - reports/model_comparison.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib

import torch
import torch.nn as nn
import torch.optim as optim


# --------------------
# Data utilities
# --------------------
def load_features(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = {"match_id", "round_num", "side", "y"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # Keep only numeric columns + required
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Ensure y is in numeric
    if "y" not in numeric_cols:
        numeric_cols.append("y")
    keep = ["match_id", "round_num", "side"] + sorted([c for c in numeric_cols if c != "y"]) + ["y"]
    return df[keep].copy()


def make_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    # Features = all numeric except y; Label = y
    num_cols = [c for c in df.columns if c not in ("match_id", "round_num", "side", "y")]
    X = df[num_cols].to_numpy(dtype=np.float32)
    y = df["y"].astype(int).to_numpy()
    return X, y, num_cols


def split_train_val_test(df: pd.DataFrame, test_size=0.15, val_size=0.15, seed=42):
    # Ensure no leakage by matching on match_id (group-wise split)
    matches = df["match_id"].unique()
    m_train, m_temp = train_test_split(matches, test_size=test_size + val_size, random_state=seed)
    rel = val_size / (test_size + val_size)
    m_val, m_test = train_test_split(m_temp, test_size=1 - rel, random_state=seed)

    def sel(ms):
        return df[df["match_id"].isin(ms)].copy()

    return sel(m_train), sel(m_val), sel(m_test)


# --------------------
# Metrics
# --------------------
def evaluate_predictions(y_true, proba):
    proba = np.clip(proba, 1e-6, 1 - 1e-6)
    pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(y_true, pred)
    try:
        auc = roc_auc_score(y_true, proba)
    except ValueError:
        auc = np.nan
    ll = log_loss(y_true, proba)
    br = brier_score_loss(y_true, proba)
    return {"Accuracy": acc, "AUC": auc, "LogLoss": ll, "Brier": br}


# --------------------
# Torch models
# --------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TinyTransformer(nn.Module):
    def __init__(self, in_dim: int, model_dim: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input = nn.Linear(in_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dim_feedforward=model_dim * 2, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos = PositionalEncoding(model_dim, max_len=512)
        self.head = nn.Sequential(nn.Linear(model_dim, 1), nn.Sigmoid())

    def forward(self, x):
        # x: (B, T, D)
        z = self.input(x)
        z = self.pos(z)
        z = self.encoder(z)  # (B, T, model_dim)
        last = z[:, -1, :]   # use last token representation
        return self.head(last).squeeze(-1)


class TDValueNet(nn.Module):
    """
    Simple TD(0) value estimator V(s). We learn to predict expected return from state.
    Here reward is y (win=1/lose=0) and gamma<1 discounts future rounds.
    """
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# --------------------
# Sequence maker
# --------------------
def make_sequences(df: pd.DataFrame, feature_cols: List[str], seq_len: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Build per-(match_id, side) sequences of length seq_len ending at each round."""
    df = df.sort_values(["match_id", "side", "round_num"]).reset_index(drop=True)
    X_seq = []
    y_seq = []
    grouped = df.groupby(["match_id", "side"], sort=False)
    for _, g in grouped:
        g = g.reset_index(drop=True)
        feat = g[feature_cols].to_numpy(dtype=np.float32)
        labels = g["y"].to_numpy(dtype=np.int64)
        for i in range(len(g)):
            start = max(0, i - seq_len + 1)
            window = feat[start:i+1]
            # pad to seq_len at the front
            if len(window) < seq_len:
                pad = np.zeros((seq_len - len(window), feat.shape[1]), dtype=np.float32)
                window = np.vstack([pad, window])
            X_seq.append(window)
            y_seq.append(labels[i])
    return np.stack(X_seq), np.array(y_seq)


# --------------------
# Training helpers
# --------------------
def train_torch_binary(model: nn.Module, X: np.ndarray, y: np.ndarray, Xv: np.ndarray, yv: np.ndarray, epochs: int = 15, lr: float = 1e-3, batch: int = 256, is_sequence: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss()

    def batches(A, B, bs):
        for i in range(0, len(A), bs):
            yield A[i:i+bs], B[i:i+bs]

    for ep in range(epochs):
        model.train()
        losses = []
        for xb, yb in batches(X, y, batch):
            xb_t = torch.from_numpy(xb).float().to(device)
            yb_t = torch.from_numpy(yb).float().to(device)
            opt.zero_grad()
            pred = model(xb_t) if not is_sequence else model(xb_t)  # both already shaped correctly
            loss = bce(pred, yb_t)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        # quick val
        model.eval()
        with torch.no_grad():
            xv_t = torch.from_numpy(Xv).float().to(device)
            pv = model(xv_t) if not is_sequence else model(xv_t)
            val_loss = bce(pv, torch.from_numpy(yv).float().to(device)).item()
        print(f"[Torch] epoch {ep+1}/{epochs} train_loss={np.mean(losses):.4f} val_loss={val_loss:.4f}")


def evaluate_torch(model: nn.Module, X: np.ndarray, y: np.ndarray) -> dict:
    device = next(model.parameters()).device
    with torch.no_grad():
        proba = model(torch.from_numpy(X).float().to(device)).cpu().numpy()
    return evaluate_predictions(y, proba)


def train_td_value(df_train: pd.DataFrame, df_val: pd.DataFrame, feature_cols: List[str], epochs: int = 10, gamma: float = 0.95) -> Tuple[TDValueNet, dict, dict]:
    X_tr, y_tr = make_xy(df_train)[0:2]
    X_va, y_va = make_xy(df_val)[0:2]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_va = scaler.transform(X_va).astype(np.float32)

    model = TDValueNet(X_tr.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCELoss()

    # Build transitions grouped by (match, side) for TD targets
    def td_targets(df):
        df = df.sort_values(["match_id", "side", "round_num"]).reset_index(drop=True)
        states = df[feature_cols].to_numpy(dtype=np.float32)
        rewards = df["y"].to_numpy(dtype=np.float32)
        # next-state value bootstrapping
        targets = np.zeros_like(rewards)
        for (m, s), g in df.groupby(["match_id", "side"], sort=False):
            idx = g.index.to_numpy()
            r = rewards[idx]
            # forward bootstrapping: target_t = r_t + gamma * V(s_{t+1}) (we'll use previous epoch's V; here we do one-pass approximation by shifting rewards)
            t = np.copy(r)
            t[:-1] = r[:-1] + gamma * r[1:]  # crude but OK for a simple baseline
            targets[idx] = t
        return states, targets

    Xs_tr, ys_tr = td_targets(df_train)
    Xs_va, ys_va = td_targets(df_val)

    Xs_tr = scaler.fit_transform(Xs_tr).astype(np.float32)
    Xs_va = scaler.transform(Xs_va).astype(np.float32)

    for ep in range(epochs):
        model.train()
        xb = torch.from_numpy(Xs_tr).to(device)
        yb = torch.from_numpy(ys_tr).to(device)
        opt.zero_grad()
        pred = model(xb)
        loss = bce(pred, yb)
        loss.backward()
        opt.step()
        with torch.no_grad():
            pv = model(torch.from_numpy(Xs_va).to(device))
            val = bce(pv, torch.from_numpy(ys_va).to(device)).item()
        print(f"[TD(0)] epoch {ep+1}/{epochs} loss={loss.item():.4f} val={val:.4f}")

    # Evaluate as a probability predictor of win
    eval_tr = evaluate_torch(model, scaler.transform(make_xy(df_train)[0]).astype(np.float32), make_xy(df_train)[1])
    eval_va = evaluate_torch(model, scaler.transform(make_xy(df_val)[0]).astype(np.float32), make_xy(df_val)[1])
    return model, eval_tr, eval_va


# --------------------
# Main
# --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True, help="CSV with engineered round/team features")
    ap.add_argument("--seq_len", type=int, default=5, help="Sequence length for Transformer")
    ap.add_argument("--epochs", type=int, default=15, help="Epochs for Torch models")
    args = ap.parse_args()

    out_models = Path("models"); out_models.mkdir(exist_ok=True, parents=True)
    out_reports = Path("reports"); out_reports.mkdir(exist_ok=True, parents=True)

    df = load_features(args.features)
    df_train, df_val, df_test = split_train_val_test(df, test_size=0.15, val_size=0.15, seed=42)
    Xtr, ytr, feat_cols = make_xy(df_train)
    Xva, yva, _ = make_xy(df_val)
    Xte, yte, _ = make_xy(df_test)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    Xte_s = scaler.transform(Xte)

    results = []

    # 1) Logistic Regression
    logi = LogisticRegression(max_iter=200)
    logi.fit(Xtr_s, ytr)
    p_tr = logi.predict_proba(Xtr_s)[:, 1]; p_va = logi.predict_proba(Xva_s)[:, 1]; p_te = logi.predict_proba(Xte_s)[:, 1]
    results.append(("LogisticRegression", "train", evaluate_predictions(ytr, p_tr)))
    results.append(("LogisticRegression", "val",   evaluate_predictions(yva, p_va)))
    results.append(("LogisticRegression", "test",  evaluate_predictions(yte, p_te)))
    joblib.dump({"model": logi, "scaler": scaler, "features": feat_cols}, out_models / "logistic.joblib")

    # 2) Random Forest
    rf = RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=42)
    rf.fit(Xtr, ytr)
    results.append(("RandomForest", "train", evaluate_predictions(ytr, rf.predict_proba(Xtr)[:, 1])))
    results.append(("RandomForest", "val",   evaluate_predictions(yva, rf.predict_proba(Xva)[:, 1])))
    results.append(("RandomForest", "test",  evaluate_predictions(yte, rf.predict_proba(Xte)[:, 1])))
    joblib.dump({"model": rf, "features": feat_cols}, out_models / "random_forest.joblib")

    # 3) Gradient Boosting
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(Xtr, ytr)
    results.append(("GradientBoosting", "train", evaluate_predictions(ytr, gb.predict_proba(Xtr)[:, 1])))
    results.append(("GradientBoosting", "val",   evaluate_predictions(yva, gb.predict_proba(Xva)[:, 1])))
    results.append(("GradientBoosting", "test",  evaluate_predictions(yte, gb.predict_proba(Xte)[:, 1])))
    joblib.dump({"model": gb, "features": feat_cols}, out_models / "grad_boost.joblib")

    # Torch models expect float32 arrays
    Xtr_f, Xva_f, Xte_f = Xtr_s.astype(np.float32), Xva_s.astype(np.float32), Xte_s.astype(np.float32)

    # 4) MLP
    mlp = MLP(in_dim=Xtr_f.shape[1], hidden=64, dropout=0.15)
    train_torch_binary(mlp, Xtr_f, ytr, Xva_f, yva, epochs=args.epochs, lr=1e-3, batch=256, is_sequence=False)
    res_tr = evaluate_torch(mlp, Xtr_f, ytr); res_va = evaluate_torch(mlp, Xva_f, yva); res_te = evaluate_torch(mlp, Xte_f, yte)
    results.append(("MLP", "train", res_tr)); results.append(("MLP", "val", res_va)); results.append(("MLP", "test", res_te))
    torch.save({"model_state": mlp.state_dict(), "in_dim": Xtr_f.shape[1]}, out_models / "mlp.pt")

    # 5) Tiny Transformer over short sequences
    # Build sequences with chosen seq_len
    seq_len = max(2, int(args.seq_len))
    Xtr_seq, ytr_seq = make_sequences(df_train, feat_cols, seq_len=seq_len)
    Xva_seq, yva_seq = make_sequences(df_val, feat_cols, seq_len=seq_len)
    Xte_seq, yte_seq = make_sequences(df_test, feat_cols, seq_len=seq_len)

    # Normalize per-feature across all sequence frames using training stats
    flat_tr = Xtr_seq.reshape(-1, Xtr_seq.shape[-1])
    mu, sigma = flat_tr.mean(axis=0, keepdims=True), flat_tr.std(axis=0, keepdims=True) + 1e-6
    Xtr_seq = ((Xtr_seq - mu) / sigma).astype(np.float32)
    Xva_seq = ((Xva_seq - mu) / sigma).astype(np.float32)
    Xte_seq = ((Xte_seq - mu) / sigma).astype(np.float32)

    trans = TinyTransformer(in_dim=Xtr_seq.shape[-1], model_dim=64, nhead=4, num_layers=2, dropout=0.1)
    train_torch_binary(trans, Xtr_seq, ytr_seq, Xva_seq, yva_seq, epochs=args.epochs, lr=1e-3, batch=128, is_sequence=True)
    res_tr = evaluate_torch(trans, Xtr_seq, ytr_seq); res_va = evaluate_torch(trans, Xva_seq, yva_seq); res_te = evaluate_torch(trans, Xte_seq, yte_seq)
    results.append(("Transformer", "train", res_tr)); results.append(("Transformer", "val", res_va)); results.append(("Transformer", "test", res_te))
    torch.save({"model_state": trans.state_dict(), "in_dim": Xtr_seq.shape[-1], "seq_len": seq_len, "mu": mu, "sigma": sigma}, out_models / "transformer.pt")

    # 6) TD(0) ValueNet — RL-flavored baseline
    td_model, res_tr, res_va = train_td_value(df_train, df_val, feat_cols, epochs=10, gamma=0.95)
    # Evaluate on test set (as a probability predictor of win)
    Xte_td, yte_td, _ = make_xy(df_test)
    scaler = StandardScaler().fit(Xtr)  # rough scaling for evaluation
    res_te = evaluate_torch(td_model, scaler.transform(Xte_td).astype(np.float32), yte_td)
    results.append(("TDValueNet", "train", res_tr)); results.append(("TDValueNet", "val", res_va)); results.append(("TDValueNet", "test", res_te))
    torch.save({"model_state": td_model.state_dict(), "in_dim": Xtr.shape[1]}, out_models / "td_value_net.pt")

    # Save comparison table
    rows = []
    for name, split, metrics in results:
        rows.append({"Model": name, "Split": split, **metrics})
    rep = pd.DataFrame(rows)
    rep.to_csv(out_reports / "model_comparison.csv", index=False)

    print("\n=== Model Comparison (head) ===")
    print(rep.head(12).to_string(index=False))
    print(f"\nSaved: {out_reports / 'model_comparison.csv'}")
    print("Saved models to:", out_models)


if __name__ == "__main__":
    main()
