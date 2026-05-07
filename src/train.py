import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

EVAL_THRESHOLD = 0.70


def get_model(model_type: str, random_state: int = 42):
    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_state)
    elif model_type == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=random_state)
    else:
        return RandomForestClassifier(random_state=random_state)


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # --- Bonus 5: Data distribution check ---
    label_dist = y_train.value_counts(normalize=True).sort_index()
    dist_info = {}
    for cls in sorted(y_train.unique()):
        pct = round(label_dist.get(cls, 0) * 100, 2)
        dist_info[f"class_{cls}_pct"] = pct
        if pct < 10:
            warnings.warn(
                f"⚠️ Class {cls} chiem {pct}% (< 10%). Du lieu co the bi lech lac!"
            )

    model_type = params.pop("model_type", "random_forest")
    random_state = params.pop("random_state", 42)
    model = get_model(model_type, random_state)

    with mlflow.start_run():
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(params)

        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"Model: {model_type} | Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # --- Bonus 3: Auto performance report ---
        cm = confusion_matrix(y_eval, preds)
        prec = precision_score(y_eval, preds, average=None)
        rec = recall_score(y_eval, preds, average=None)

        os.makedirs("outputs", exist_ok=True)
        report_lines = [
            "=== BAO CAO HIEU SUAT MO HINH ===",
            f"Model type: {model_type}",
            f"Accuracy: {acc:.4f}",
            f"F1-score (weighted): {f1:.4f}",
            "",
            "--- Confusion Matrix ---",
        ]
        for row in cm:
            report_lines.append("  ".join(str(int(x)) for x in row))

        report_lines.append("")
        report_lines.append("--- Precision & Recall per class ---")
        for i, (p, r) in enumerate(zip(prec, rec)):
            report_lines.append(f"Class {i}: Precision={p:.4f}, Recall={r:.4f}")

        report = "\n".join(report_lines)
        print(report)

        with open("outputs/report.txt", "w") as f:
            f.write(report)

        # --- Bonus 5: Write distribution to metrics.json ---
        metrics_out = {
            "accuracy": acc,
            "f1_score": f1,
            "label_distribution": dist_info,
        }
        with open("outputs/metrics.json", "w") as f:
            json.dump(metrics_out, f, indent=2)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
