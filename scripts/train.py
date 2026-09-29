"""Train the sign encoder.  python -m scripts.train [--split block|random] [--epochs 60]"""
import argparse

from signvla.encoder.train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["block", "random"], default="block",
                    help="block = hold out later recordings per class (recommended, harder)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    train(a.split, a.epochs, a.batch_size, a.lr, a.hidden_dim, seed=a.seed)


if __name__ == "__main__":
    main()
