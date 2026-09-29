"""Train the Bi-GRU sign encoder. Model selection uses VAL only; TEST is scored once at the end."""
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from signvla import config
from signvla.encoder.data import load_dataset
from signvla.encoder.model import GestureGRU


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _loader(X, y, batch_size, shuffle):
    return DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)), batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict(model, X, device="cpu"):
    model.eval()
    return model(torch.tensor(X).to(device)).argmax(1).cpu().numpy()


def train(strategy="block", epochs=60, batch_size=16, lr=1e-3, hidden_dim=128,
          noise=0.01, seed=42, out=config.MODEL_PATH):
    set_seed(seed)
    d = load_dataset(strategy)
    gestures = d["gestures"]
    device = torch.device("cpu")  # tiny model; CPU is deterministic and fast enough
    model = GestureGRU(len(gestures), hidden_dim=hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = _loader(d["X_train"], d["y_train"], batch_size, True)

    print(f"split={strategy}  train={len(d['y_train'])} val={len(d['y_val'])} test={len(d['y_test'])}")
    best_val, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb + noise * torch.randn_like(xb) * (xb != 0)  # jitter real landmarks only
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        sched.step()
        val_acc = float((predict(model, d["X_val"]) == d["y_val"]).mean())
        if val_acc > best_val:  # strict '>' keeps the earliest best epoch
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == epochs:
            print(f"epoch {epoch:3d}  val_acc {val_acc * 100:.1f}%  (best {best_val * 100:.1f}%)")

    model.load_state_dict(best_state)
    test_acc = float((predict(model, d["X_test"]) == d["y_test"]).mean())
    config.MODELS_DIR.mkdir(exist_ok=True)
    torch.save({
        "model_state_dict": best_state, "num_classes": len(gestures), "hidden_dim": hidden_dim,
        "gestures": gestures, "feature_version": config.FEATURE_VERSION,
        "split_strategy": strategy, "val_acc": best_val, "test_acc": test_acc,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, str(out))
    print(f"\nBest val acc {best_val * 100:.2f}% | TEST acc {test_acc * 100:.2f}% (split={strategy})")
    print(f"Saved {out}")
    return best_val, test_acc
