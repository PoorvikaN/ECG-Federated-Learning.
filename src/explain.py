import matplotlib.pyplot as plt
import shap
import torch
import numpy as np

from .config import RESULTS_DIR
from .train_baseline import run_baseline


class PredictWrapper:
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        x_tensor = torch.tensor(np.array(x), dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs


def run_explainability():
    model, X_train, X_test, _, _ = run_baseline()

    background = X_train[:50]
    samples = X_test[:100]

    wrapped_model = PredictWrapper(model)
    explainer = shap.Explainer(wrapped_model, background)
    shap_values = explainer(samples)

    out_path = RESULTS_DIR / "shap_summary.png"

    plt.figure()
    shap.summary_plot(shap_values, samples, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved SHAP summary plot to: {out_path}")
