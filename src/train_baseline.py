import json
from sklearn.metrics import classification_report

from .config import RESULTS_DIR
from .data_utils import get_train_test_data
from .model import ECGModel, train_model, evaluate_model


def run_baseline():
    X_train, X_test, y_train, y_test, _, label_encoder = get_train_test_data()
    num_classes = len(label_encoder.classes_)

    model = ECGModel(input_dim=X_train.shape[1], num_classes=num_classes)
    model = train_model(model, X_train, y_train, epochs=10, lr=1e-3)

    acc, preds = evaluate_model(model, X_test, y_test)

    report = classification_report(y_test, preds, output_dict=True, zero_division=0)

    output = {
        "accuracy": acc,
        "classes": label_encoder.classes_.tolist(),
        "report": report,
    }

    out_file = RESULTS_DIR / "baseline_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Baseline accuracy: {acc:.4f}")
    print(f"Saved metrics to: {out_file}")

    return model, X_train, X_test, y_test, label_encoder
