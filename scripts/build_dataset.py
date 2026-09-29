"""Compile data/landmarks/*/*.npy into data/dataset.npz (+ label_map.json).

Usage: python -m scripts.build_dataset
"""
import numpy as np

from signvla.encoder.data import compile_dataset, load_dataset


def main():
    X, y, names = compile_dataset()
    print(f"Compiled {len(X)} sequences, {len(names)} classes, X shape {X.shape}")
    for strategy in ("block", "random"):
        d = load_dataset(strategy)
        print(f"  split={strategy:<6} train {len(d['y_train'])}  val {len(d['y_val'])}  test {len(d['y_test'])}")
    print("Class counts:", {n: int((y == i).sum()) for i, n in enumerate(names)})


if __name__ == "__main__":
    main()
