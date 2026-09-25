"""
ISL Gesture Model Training Script
Trains a Bi-directional GRU model on landmark sequence dataset (dataset.npz)
and saves the trained weights to models/isl_gesture_model.pth.
"""
import os
import sys
import json
import argparse
from pathlib import Path

# Auto re-exec inside virtual environment if invoked with global system Python
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VENV_PY = _PROJECT_ROOT / ".venv" / "bin" / "python"
if not _VENV_PY.exists():
    _VENV_PY = _PROJECT_ROOT / "venv" / "bin" / "python"

if _VENV_PY.exists() and sys.prefix == sys.base_prefix:
    os.execv(str(_VENV_PY), [str(_VENV_PY)] + sys.argv)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(str(_PROJECT_ROOT))
import config
from src.model import get_model


def train(epochs=40, batch_size=16, lr=1e-3, hidden_dim=128):
    print("\n" + "=" * 60)
    print("  ISL GESTURE MODEL TRAINING")
    print("=" * 60)

    dataset_path = config.DATA_DIR / "dataset.npz"
    label_map_path = config.DATA_DIR / "label_map.json"

    if not dataset_path.exists() or not label_map_path.exists():
        print("[ERROR] Dataset not found! Run src/create_dataset.py first.")
        return

    # Load data
    data = np.load(str(dataset_path))
    X_train = torch.tensor(data["X_train"], dtype=torch.float32)
    y_train = torch.tensor(data["y_train"], dtype=torch.long)
    X_test = torch.tensor(data["X_test"], dtype=torch.float32)
    y_test = torch.tensor(data["y_test"], dtype=torch.long)

    with open(label_map_path, "r") as f:
        meta = json.load(f)
    gestures = meta["gestures"]
    num_classes = len(gestures)

    print(f"Loaded dataset:")
    print(f"  Train samples: {len(X_train)}  |  Shape: {X_train.shape}")
    print(f"  Test samples : {len(X_test)}  |  Shape: {X_test.shape}")
    print(f"  Classes ({num_classes}): {gestures}\n")

    # DataLoader
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Model, Loss, Optimizer
    model = get_model(num_classes=num_classes, input_dim=config.TOTAL_FEATURES_PER_FRAME, hidden_dim=hidden_dim)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_weights_path = config.MODELS_DIR / "isl_gesture_model.pth"

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

        scheduler.step()
        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_x.size(0)

        val_acc = val_correct / max(val_total, 1)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "input_dim": config.TOTAL_FEATURES_PER_FRAME,
                "hidden_dim": hidden_dim,
                "gestures": gestures,
                "val_acc": val_acc
            }, str(best_weights_path))

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% || "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}%")

    print("\n" + "-" * 60)
    print(f"Training completed! Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Model saved to: {best_weights_path}")
    print("-" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ISL Gesture recognition model.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, hidden_dim=args.hidden_dim)
