"""
Train / evaluate the deep-learning and transformer models.

  python -m src.deep_learning.train --model transformer
  python -m src.deep_learning.train --model all --epochs 40
  python -m src.deep_learning.train --model transformer --smoke      # 1-minute wiring check

Outputs (per model):
  models/deep_learning/<name>.pt                  weights + config + scaler (everything needed to predict)
  reports/deep_learning/<name>_metrics.json       test metrics
  reports/deep_learning/<name>_history.csv        loss / validation curve per epoch
  reports/deep_learning/predictions/<name>.csv    test-period predictions
Then run  python -m src.deep_learning.report  to build the comparison table and charts.
"""
import argparse
import copy
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import config as C
from .data import prepare
from .models import MODEL_NAMES, build_model, count_parameters
from .results_io import save_results


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, X: np.ndarray, device, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out)


def val_mae_vehicles(model, split, device) -> float:
    pred = np.clip(predict(model, split.X, device), 0.0, 1.25)
    return float(np.mean(np.abs(pred - split.y) * split.capacity))


def train_one(model_name: str, args, splits, scaler, features):
    tr, va, te = splits["train"], splits["val"], splits["test"]
    if args.smoke:                                   # tiny subset, just to check everything is wired up
        tr.X, tr.y = tr.X[:2048], tr.y[:2048]
        va.X, va.y, va.capacity = va.X[:512], va.y[:512], va.capacity[:512]

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(args.seed)
    kwargs = dict(hidden=args.hidden, d_model=args.d_model, layers=args.layers,
                  heads=args.heads, dropout=args.dropout)
    model = build_model(model_name, len(features), args.seq_len, **kwargs).to(device)
    n_params = count_parameters(model)
    name = f"{model_name}_smoke" if args.smoke else model_name
    print(f"\n=== {name} | {n_params:,} parameters | device={device} | "
          f"train/val/test windows = {len(tr.X):,}/{len(va.X):,}/{len(te.X):,}")

    loader = DataLoader(TensorDataset(torch.from_numpy(tr.X), torch.from_numpy(tr.y)),
                        batch_size=args.batch_size, shuffle=True)
    loss_fn = nn.HuberLoss(delta=0.1)                # robust to the rare rush-hour spikes
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

    best_mae, best_state, best_epoch, bad_epochs = float("inf"), None, 0, 0
    history, t0 = [], time.time()
    epochs = 2 if args.smoke else args.epochs
    for epoch in range(1, epochs + 1):
        model.train()
        running, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * len(xb)
            n += len(xb)
        v_mae = val_mae_vehicles(model, va, device)
        sched.step(v_mae)
        lr_now = opt.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": running / n, "val_MAE_vehicles": v_mae, "lr": lr_now})
        flag = ""
        if v_mae < best_mae - 1e-4:
            best_mae, best_epoch, bad_epochs = v_mae, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            flag = "  *best*"
        else:
            bad_epochs += 1
        print(f"epoch {epoch:3d}  train_loss={running / n:.5f}  val_MAE={v_mae:6.2f} veh  lr={lr_now:.1e}{flag}")
        if bad_epochs >= args.patience:
            print(f"early stopping (no improvement for {args.patience} epochs)")
            break

    if best_state is None:                           # only if every validation score was NaN
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    pred = predict(model, te.X[:512] if args.smoke else te.X, device)
    te_eval = te
    if args.smoke:
        te_eval = type(te)(te.X[:512], te.y[:512], te.capacity[:512], te.timestamp[:512], te.site[:512], te.y_class[:512])
    payload = save_results(name, te_eval, pred, history=history, extra={
        "n_parameters": n_params, "best_epoch": best_epoch, "epochs_run": len(history),
        "train_seconds": round(time.time() - t0, 1), "device": str(device),
        "seq_len": args.seq_len, "hyperparameters": {**kwargs, "lr": args.lr, "batch_size": args.batch_size}})

    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_name": model_name, "state_dict": best_state, "model_kwargs": kwargs,
                "seq_len": args.seq_len, "features": features,
                "scaler_mean": scaler["mean"].tolist(), "scaler_scale": scaler["scale"].tolist(),
                "class_names": C.CLASS_NAMES, "util_bins": C.UTIL_BINS}, C.MODEL_DIR / f"{name}.pt")
    m = payload["metrics"]
    print(f"TEST  MAE={m['MAE_vehicles']:.2f} veh  RMSE={m['RMSE_vehicles']:.2f}  R2={m['R2']:.4f}  "
          f"class_acc={m['class_accuracy']:.4f}  macroF1={m['class_macro_F1']:.4f}")
    return payload


def main():
    ap = argparse.ArgumentParser(description="Train deep-learning / transformer traffic models")
    ap.add_argument("--model", default="transformer", choices=MODEL_NAMES + ["all"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seq-len", type=int, default=C.SEQ_LEN)
    ap.add_argument("--hidden", type=int, default=64, help="RNN width (MLP uses 4x)")
    ap.add_argument("--d-model", type=int, default=64, help="transformer width")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    ap.add_argument("--smoke", action="store_true", help="tiny 2-epoch run to check the code works")
    args = ap.parse_args()

    names = MODEL_NAMES if args.model == "all" else [args.model]
    for name in names:
        splits, scaler, features = prepare(args.seq_len)     # fresh copy each time (smoke mode slices in place)
        train_one(name, args, splits, scaler, features)


if __name__ == "__main__":
    main()
