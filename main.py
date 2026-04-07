import argparse

from src.train_baseline import run_baseline
from src.explain import run_explainability
from src.federated_simulation import run_federated


def main():
    parser = argparse.ArgumentParser(description="ECG Explainable Federated Learning")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["baseline", "explain", "federated"],
        help="Execution mode",
    )
    args = parser.parse_args()

    if args.mode == "baseline":
        run_baseline()
    elif args.mode == "explain":
        run_explainability()
    elif args.mode == "federated":
        run_federated()


if __name__ == "__main__":
    main()
