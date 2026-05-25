#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Polazni kod treba samo da se promeni da radi nad mojim CSV-om, a sintetička od demoa izbaciti. 

Razumeo. Pravilo za sve buduće modele:

polazni kod iz članka se direktno menja da radi nad tvojim loto CSV-om
sintetička demo data, neiskorišćeni delovi, sve što ne pripada polaznom zadatku se izbacuje
predviđa se sledeće loto kolo + back-test, snimanje u TXT
bez paralelnih "novih" klasa ispod polaznog, bez pitanja




Jasno. Ponavljam svojim rečima da budem siguran:

Polazni kod iz članka prvo mora da proradi kakav jeste (sintetički demo, ništa ne ruši).
Tek posle se polazni direktno menja da radi nad tvojim loto CSV-om — sintetička demo data se izbacuje, svi neiskorišćeni delovi se izbacuju.
Cilj v2 fajla: predviđa sledeće loto kolo + back-test, snimanje rezultata u TXT.
Bez paralelnih „novih" klasa ispod polaznog, bez postavljanja pitanja kad je sve jasno.
"""




"""
Hibridne arhitekture za predikciju koje kombinuju deep learning i klasične time-series modele.

7. Spacetimeformer: Physics-Informed Attention (Physics-Informed Loss)


Loto pipeline: CSV → multihot + rolling + gap + statistike → standardizacija → train/val/back-test (200/100).
physics_params per primer = prosek standardizovanih statistika kola (suma, neparnih, niskih, raspon) preko LOOK_BACK prozora — domenske konstante u loto kontekstu, analogno [wind_speed, density, efficiency].
coordinates se ne koristi u polaznom forward-u, šaljem nule.
BCEWithLogitsLoss sa pos_weight=(N_MAX-K)/K, (best/final/ensemble + back-test 100 + TXT izlaz).
Parametri: LOOK_BACK=128, EPOCHS=50, BATCH=64, EMBED_DIM=128, N_LAYERS=4.

Spacetimeformer sa embed_dim=128, n_layers=4, LOOK_BACK=128 može biti sporiji 
"""


import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

class PhysicsInformedAttention(nn.Module):
    def __init__(self, embed_dim, physics_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        
        # Physics embedding: encodes domain equations
        self.physics_encoder = nn.Sequential(
            nn.Linear(physics_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, query, key, value, physics_params):
        """
        Args:
            physics_params: [batch, physics_dim] e.g., diffusion coefficient, viscosity
        """
        # Encode physics as bias in attention scores
        physics_bias = self.physics_encoder(physics_params)
        physics_bias = physics_bias.unsqueeze(1)  # [batch, 1, embed_dim]
        
        # Standard attention with physics-informed bias
        attn_output, attn_weights = self.attention(
            query + physics_bias, key + physics_bias, value
        )
        return attn_output, attn_weights

class Spacetimeformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_projection = nn.Linear(config['input_dim'], config['embed_dim'])
        self.physics_attn = PhysicsInformedAttention(
            embed_dim=config['embed_dim'],
            physics_dim=config['physics_dim']
        )
        
        # Transformer encoder with physics-informed layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config['embed_dim'],
                nhead=8,
                dim_feedforward=512,
                dropout=0.1,
                batch_first=True
            ) for _ in range(config['n_layers'])
        ])
        
        self.forecast_head = nn.Linear(config['embed_dim'], config['forecast_len'])
        
    def physics_residual_loss(self, predictions, physics_params):
        """
        Enforce PDE constraints: e.g., diffusion equation du/dt = alpha * d²u/dx²
        """
        # Compute temporal gradient (simplified finite difference)
        pred_t = predictions  # [batch, seq_len]
        dt = 1.0  # Time step
        
        # Compute spatial gradient (requires grid coordinates)
        # For wind farm: enforce power output cannot exceed Betz limit
        betz_limit = 16/27  # Max theoretical efficiency
        
        # Penalty for predictions exceeding physical limit
        violation = torch.relu(predictions - betz_limit)
        physics_loss = violation.mean()
        
        return physics_loss
    
    def forward(self, x, physics_params, coordinates):
        """
        Args:
            x: [batch, seq_len, features]
            physics_params: [batch, physics_dim] domain constants
            coordinates: [batch, seq_len, 2] spatial coordinates for PDE
        """
        if physics_params.size(0) == 1 and x.size(0) > 1:
            physics_params = physics_params.expand(x.size(0), -1)

        x = self.input_projection(x)

        # Physics-informed attention
        attn_out, attn_map = self.physics_attn(x, x, x, physics_params)
        
        # Standard transformer layers
        for layer in self.layers:
            attn_out = layer(attn_out)
            
        # Global pooling for forecasting
        pooled = attn_out.mean(dim=1)
        forecast = self.forecast_head(pooled)
        
        return forecast, attn_map
    
    def hybrid_loss(self, predictions, targets, physics_params, lambda_physics=0.5):
        """
        Combines NLL loss with physics residual penalty
        """
        # Data loss
        data_loss = nn.MSELoss()(predictions, targets)
        
        # Physics loss
        physics_loss = self.physics_residual_loss(predictions, physics_params)
        
        return data_loss + lambda_physics * physics_loss, data_loss, physics_loss

# =========================
# Loto 7/39 adaptacija (loto7hh_4620_k41.csv) — demo izbačen
# =========================
import os

SEED = 39
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import copy
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from sklearn.metrics import label_ranking_average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


CSV_PATH = "/Users/4c/Desktop/GHQ/KvantniRegresor/loto7hh_4620_k41.csv"
OUT_TXT = Path("/Users/4c/Desktop/GHQ/TimeSeriesModels/7_Spacetimeformer_loto_v2_predikcija.txt")

N_MIN, N_MAX = 1, 39
K = 7
LOOK_BACK = 128
WINDOWS_RF = (20, 50, 100)
BACKTEST_N = 100
VAL_N = 200
EPOCHS = 50
BATCH = 64
LR = 1e-3
EMBED_DIM = 128
N_LAYERS = 4
PHYSICS_DIM = 4  # suma, neparnih, niskih, raspon (domenske "konstante" loto kola)

T0 = time.time()
print()
print("START 7_Spacetimeformer_loto_v2", datetime.today())
print()

df = pd.read_csv(CSV_PATH).iloc[:, :K].astype(int)
draws = np.sort(df.values, axis=1)
N_total = draws.shape[0]
if not ((draws >= N_MIN) & (draws <= N_MAX)).all():
    raise ValueError("CSV ima brojeve van opsega 1..39.")
for idx, row in enumerate(draws):
    if len(set(row.tolist())) != K:
        raise ValueError(f"Red {idx} nema 7 jedinstvenih brojeva: {row.tolist()}")

print(f"CSV: {CSV_PATH}")
print(f"Broj izvlačenja: {N_total}, brojeva po kolu: {K}")
print()


def draws_to_multihot(rows):
    out = np.zeros((rows.shape[0], N_MAX), dtype=np.float32)
    for i, row in enumerate(rows):
        out[i, row - 1] = 1.0
    return out


def rolling_features(y_multi):
    cum = np.cumsum(y_multi, axis=0)
    blocks = []
    for w in WINDOWS_RF:
        rolled = np.zeros_like(cum, dtype=np.float32)
        rolled[1:w + 1] = cum[:w]
        rolled[w + 1:] = cum[w:-1] - cum[:-w - 1]
        blocks.append(rolled / float(w))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def gap_matrix(rows):
    n = rows.shape[0]
    gap = np.zeros((n, N_MAX), dtype=np.float32)
    last_seen = np.full(N_MAX, -1, dtype=int)
    for i, row in enumerate(rows):
        for k in range(N_MAX):
            gap[i, k] = (i - last_seen[k]) if last_seen[k] >= 0 else i + 1
        for v in row:
            last_seen[v - 1] = i
    return gap


def make_sequences(features, targets, physics_steps, look_back):
    X, Y, P = [], [], []
    for i in range(look_back, len(features)):
        X.append(features[i - look_back:i])
        Y.append(targets[i])
        # physics_params per primer: prosek standardizovanih stats kola u prozoru
        P.append(physics_steps[i - look_back:i].mean(axis=0))
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(Y, dtype=np.float32),
        np.asarray(P, dtype=np.float32),
    )


def topk_from_scores(scores_1d, k=K):
    s = np.asarray(scores_1d, dtype=float)
    order = np.lexsort((np.arange(N_MAX), -s))
    return np.sort(order[:k] + 1)


def avg_hits(scores_2d, y_true):
    hits = 0
    for i in range(scores_2d.shape[0]):
        true_set = set(np.where(y_true[i] == 1)[0] + 1)
        pred_set = set(topk_from_scores(scores_2d[i]).tolist())
        hits += len(true_set & pred_set)
    return hits / scores_2d.shape[0]


def safe_auc(y_true, scores):
    try:
        return roc_auc_score(y_true, scores, average="macro")
    except Exception:
        return float("nan")


def safe_lrap(y_true, scores):
    try:
        return label_ranking_average_precision_score(y_true.astype(int), scores)
    except Exception:
        return float("nan")


def describe(pick):
    return (
        f"suma={int(pick.sum())}, "
        f"neparnih={int((pick % 2 == 1).sum())}/{K}, "
        f"niskih(<=19)={int((pick <= 19).sum())}/{K}, "
        f"raspon={int(pick.max() - pick.min())}"
    )


Y_full = draws_to_multihot(draws)
rolling_raw = rolling_features(Y_full)
gap_raw = gap_matrix(draws)

sum_col = draws.sum(axis=1, keepdims=True).astype(np.float32)
odd_col = (draws % 2 == 1).sum(axis=1, keepdims=True).astype(np.float32)
low_col = (draws <= 19).sum(axis=1, keepdims=True).astype(np.float32)
range_col = (draws.max(axis=1, keepdims=True) - draws.min(axis=1, keepdims=True)).astype(np.float32)
stats_raw = np.concatenate([sum_col, odd_col, low_col, range_col], axis=1)  # [T, 4] = physics ulaz

step_features_raw = np.concatenate([Y_full, rolling_raw, gap_raw, stats_raw], axis=1).astype(np.float32)

START = max(LOOK_BACK, max(WINDOWS_RF))
feature_scaler = StandardScaler()
step_features = step_features_raw.copy()
step_features[START:] = feature_scaler.fit_transform(step_features_raw[START:]).astype(np.float32)
step_features[:START] = feature_scaler.transform(step_features_raw[:START]).astype(np.float32)

# Physics stats — standardizovane na isti način, koriste se kao physics_params
physics_scaler = StandardScaler()
physics_steps = stats_raw.copy().astype(np.float32)
physics_steps[START:] = physics_scaler.fit_transform(stats_raw[START:]).astype(np.float32)
physics_steps[:START] = physics_scaler.transform(stats_raw[:START]).astype(np.float32)

X_seq, Y_seq, P_seq = make_sequences(step_features, Y_full, physics_steps, LOOK_BACK)
X_seq = X_seq[START - LOOK_BACK:]
Y_seq = Y_seq[START - LOOK_BACK:]
P_seq = P_seq[START - LOOK_BACK:]

n_total = X_seq.shape[0]
n_train = n_total - BACKTEST_N
assert n_train > VAL_N + 200, "Premalo podataka za train/val/back-test."

X_tr, Y_tr, P_tr = X_seq[:n_train - VAL_N], Y_seq[:n_train - VAL_N], P_seq[:n_train - VAL_N]
X_val, Y_val, P_val = X_seq[n_train - VAL_N:n_train], Y_seq[n_train - VAL_N:n_train], P_seq[n_train - VAL_N:n_train]
X_back, Y_back, P_back = X_seq[n_train:], Y_seq[n_train:], P_seq[n_train:]

X_next = step_features[-LOOK_BACK:].reshape(1, LOOK_BACK, step_features.shape[1]).astype(np.float32)
P_next = physics_steps[-LOOK_BACK:].mean(axis=0, keepdims=True).astype(np.float32)

INPUT_DIM = X_seq.shape[-1]
print(f"Feature dim: {INPUT_DIM}, LOOK_BACK: {LOOK_BACK}, physics_dim: {PHYSICS_DIM}")
print(f"Train: {X_tr.shape[0]}, Val: {X_val.shape[0]}, Back-test: {X_back.shape[0]}")
print()


config = {
    'input_dim': INPUT_DIM,
    'embed_dim': EMBED_DIM,
    'physics_dim': PHYSICS_DIM,
    'n_layers': N_LAYERS,
    'forecast_len': N_MAX  # 39 sigmoid logita po broju 1..39
}

model = Spacetimeformer(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

pos_weight_value = (N_MAX - K) / K
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.full((N_MAX,), pos_weight_value, dtype=torch.float32))


def zero_coords(batch_size):
    # Spacetimeformer.forward zahteva coordinates argument, ali ga ne koristi unutra; šaljem nule
    return torch.zeros(batch_size, LOOK_BACK, 2, dtype=torch.float32)


def make_loader(X, P, Y, shuffle):
    generator = torch.Generator()
    generator.manual_seed(SEED)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(P), torch.from_numpy(Y))
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, generator=generator)


train_loader = make_loader(X_tr, P_tr, Y_tr, shuffle=False)
val_X_t = torch.from_numpy(X_val)
val_P_t = torch.from_numpy(P_val)
val_Y_t = torch.from_numpy(Y_val)
val_coords = zero_coords(X_val.shape[0])

best_state = copy.deepcopy(model.state_dict())
best_val_loss = float("inf")
best_epoch = 0

print("Treniranje Spacetimeformer na loto podacima ...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    seen = 0
    for xb, pb, yb in train_loader:
        optimizer.zero_grad(set_to_none=True)
        coords = zero_coords(xb.size(0))
        logits, _ = model(xb, pb, coords)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += float(loss.detach().cpu()) * xb.size(0)
        seen += xb.size(0)
    train_loss /= max(seen, 1)

    model.eval()
    with torch.no_grad():
        val_logits, _ = model(val_X_t, val_P_t, val_coords)
        val_loss = float(criterion(val_logits, val_Y_t).detach().cpu())
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())

    if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
        print(f"epoch {epoch:4d}/{EPOCHS}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  best_epoch={best_epoch}")

final_state = copy.deepcopy(model.state_dict())
print()
print(f"✅ Trening završen. best_epoch={best_epoch}, best_val_loss={best_val_loss:.5f}")
print()


def predict_scores(model, X, P):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, X.shape[0], BATCH):
            xb = torch.from_numpy(X[s:s + BATCH])
            pb = torch.from_numpy(P[s:s + BATCH])
            coords = zero_coords(xb.size(0))
            logits, _ = model(xb, pb, coords)
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(out)


def evaluate(model, X, P, Y):
    scores = predict_scores(model, X, P)
    return scores, avg_hits(scores, Y), safe_auc(Y, scores), safe_lrap(Y, scores)


model.load_state_dict(best_state)
scores_best, h_best, auc_best, lrap_best = evaluate(model, X_back, P_back, Y_back)
next_best = predict_scores(model, X_next, P_next)[0]
pick_best = topk_from_scores(next_best)

model.load_state_dict(final_state)
scores_final, h_final, auc_final, lrap_final = evaluate(model, X_back, P_back, Y_back)
next_final = predict_scores(model, X_next, P_next)[0]
pick_final = topk_from_scores(next_final)

ensemble_scores = (scores_best + scores_final) / 2.0
h_ens = avg_hits(ensemble_scores, Y_back)
auc_ens = safe_auc(Y_back, ensemble_scores)
lrap_ens = safe_lrap(Y_back, ensemble_scores)
pick_ens = topk_from_scores((next_best + next_final) / 2.0)

for name, pick in [("STF_best", pick_best), ("STF_final", pick_final), ("STF_ensemble", pick_ens)]:
    assert len(set(pick.tolist())) == K, f"{name} nema 7 jedinstvenih brojeva"
    assert pick.min() >= N_MIN and pick.max() <= N_MAX, f"{name} van opsega"
    assert list(pick) == sorted(pick.tolist()), f"{name} nije sortiran"

print("Predikcija sledeće Loto 7/39 kombinacije:")
print(f"Spacetimeformer_best     -> {pick_best.tolist()}  ({describe(pick_best)})")
print(f"Spacetimeformer_final    -> {pick_final.tolist()}  ({describe(pick_final)})")
print(f"Spacetimeformer_ensemble -> {pick_ens.tolist()}  ({describe(pick_ens)})")
print()

print("Back-test (poslednjih 100 izvlačenja):")
print(f"{'model':<26} {'hits/7':>8} {'hit%':>7} {'AUC':>7} {'LRAP':>7}")
print(f"{'Spacetimeformer_best':<26} {h_best:>8.3f} {100*h_best/K:>6.1f}% {auc_best:>7.3f} {lrap_best:>7.3f}")
print(f"{'Spacetimeformer_final':<26} {h_final:>8.3f} {100*h_final/K:>6.1f}% {auc_final:>7.3f} {lrap_final:>7.3f}")
print(f"{'Spacetimeformer_ensemble':<26} {h_ens:>8.3f} {100*h_ens/K:>6.1f}% {auc_ens:>7.3f} {lrap_ens:>7.3f}")
print(f"(slučajan baseline ≈ {7*7/39:.3f} hits/7)")
print()


elapsed = time.time() - T0
with OUT_TXT.open("a", encoding="utf-8") as f:
    f.write(f"\n--- {datetime.today()} (seed={SEED}, N={N_total}, epochs={EPOCHS}) ---\n")
    f.write(f"Spacetimeformer_best     -> {pick_best.tolist()}  ({describe(pick_best)})\n")
    f.write(f"Spacetimeformer_final    -> {pick_final.tolist()}  ({describe(pick_final)})\n")
    f.write(f"Spacetimeformer_ensemble -> {pick_ens.tolist()}  ({describe(pick_ens)})\n")
    f.write(
        f"back-test: BEST hits/7={h_best:.3f}, AUC={auc_best:.3f}, LRAP={lrap_best:.3f}; "
        f"FINAL hits/7={h_final:.3f}, AUC={auc_final:.3f}, LRAP={lrap_final:.3f}; "
        f"ENSEMBLE hits/7={h_ens:.3f}, AUC={auc_ens:.3f}, LRAP={lrap_ens:.3f}; "
        f"baseline={7*7/39:.3f}\n"
    )
    f.write(f"elapsed={elapsed:.1f}s\n")

print(f"Snimljeno u: {OUT_TXT}")
print()
print("STOP", datetime.today())
print(f"Ukupno vreme: {str(timedelta(seconds=int(elapsed)))}  ({elapsed:.1f} s)")

"""
START 7_Spacetimeformer_loto_v2 2026-05-25 17:15:46.190236

CSV: /loto7hh_4620_k41.csv
Broj izvlačenja: 4620, brojeva po kolu: 7

Feature dim: 199, LOOK_BACK: 128, physics_dim: 4
Train: 4192, Val: 200, Back-test: 100

Treniranje Spacetimeformer na loto podacima ...
epoch    1/50  train_loss=1.15046  val_loss=1.14229  best_epoch=1
epoch   10/50  train_loss=1.13610  val_loss=1.14916  best_epoch=3
epoch   20/50  train_loss=1.12233  val_loss=1.16123  best_epoch=3
epoch   30/50  train_loss=1.07342  val_loss=1.22559  best_epoch=3
epoch   40/50  train_loss=0.99772  val_loss=1.29367  best_epoch=3
epoch   50/50  train_loss=0.94043  val_loss=1.37282  best_epoch=3

✅ Trening završen. best_epoch=3, best_val_loss=1.14046

Predikcija sledeće Loto 7/39 kombinacije:
Spacetimeformer_best     -> [8, 10, 11, 23, 32, 33, 34]  (suma=151, neparnih=3/7, niskih(<=19)=3/7, raspon=26)
Spacetimeformer_final    -> [3, 7, 16, 17, 30, 31, 37]  (suma=141, neparnih=5/7, niskih(<=19)=4/7, raspon=34)
Spacetimeformer_ensemble -> [7, 16, 17, 23, 30, 31, 37]  (suma=161, neparnih=5/7, niskih(<=19)=3/7, raspon=30)

Back-test (poslednjih 100 izvlačenja):
model                        hits/7    hit%     AUC    LRAP
Spacetimeformer_best          1.290   18.4%   0.525   0.249
Spacetimeformer_final         1.210   17.3%   0.524   0.247
Spacetimeformer_ensemble      1.240   17.7%   0.525   0.247
(slučajan baseline ≈ 1.256 hits/7)

Snimljeno u: /7_Spacetimeformer_loto_v2_predikcija.txt

STOP 2026-05-25 17:49:39.990337
Ukupno vreme: 0:33:53  (2033.8 s)
"""
