# modelbaru/train.py
import os
import json
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report, mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def infer_task(df: pd.DataFrame, target: str) -> str:
    """Auto-deteksi task: classification vs regression."""
    y = df[target]
    if pd.api.types.is_numeric_dtype(y):
        # banyak unique -> kemungkinan regresi
        if y.nunique() > max(20, int(0.05 * len(y))):
            return "regression"
        # numeric tapi sedikit kelas → treat as classification
        return "classification"
    return "classification"


def build_model(task: str, model_name: str, args):
    if task == "classification":
        if model_name == "logreg":
            return LogisticRegression(max_iter=1000, n_jobs=None)
        elif model_name == "rf":
            return RandomForestClassifier(
                n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=args.random_state
            )
        else:
            raise ValueError(f"Unknown classification model: {model_name}")
    elif task == "regression":
        if model_name == "linear":
            return LinearRegression()
        elif model_name == "rf":
            return RandomForestRegressor(
                n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=args.random_state
            )
        else:
            raise ValueError(f"Unknown regression model: {model_name}")
    else:
        raise ValueError(f"Unknown task: {task}")


def plot_confusion_matrix(cm, labels, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train model with MLflow logging")
    parser.add_argument("--data_csv", type=str, default="data/dataset.csv",
                        help="Path ke CSV data (harus termasuk kolom target).")
    parser.add_argument("--target", type=str, required=True, help="Nama kolom target.")
    parser.add_argument("--task", type=str, default="auto",
                        choices=["auto", "classification", "regression"],
                        help="Jenis task. 'auto' akan infer dari data.")
    parser.add_argument("--model", type=str, default="rf",
                        help="Model: classification: [rf|logreg]; regression: [rf|linear]")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--experiment", type=str, default=os.getenv("MLFLOW_EXPERIMENT", "default"))
    parser.add_argument("--registered_model_name", type=str, default=os.getenv("MLFLOW_REGISTERED_MODEL", ""))

    args = parser.parse_args()

    # MLflow tracking URI (ambil dari env, fallback ke localhost)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    # Load data
    data_path = Path(args.data_csv)
    if not data_path.exists():
        # fallback: generate dummy dataset biar gak crash
        print(f"[WARN] File {data_path} tidak ditemukan. Membuat dummy dataset klasifikasi...")
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=500, n_features=8, n_informative=5,
                                   n_redundant=1, n_classes=2, random_state=args.random_state)
        df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(X.shape[1])])
        df["target"] = y
        args.target = "target"
        if args.task == "auto":
            args.task = "classification"
    else:
        df = pd.read_csv(data_path)

    if args.task == "auto":
        args.task = infer_task(df, args.target)

    # Split features/target
    y = df[args.target]
    X = df.drop(columns=[args.target])

    # Identify numeric/categorical
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    model = build_model(args.task, args.model, args)

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y if args.task == "classification" else None
    )

    with mlflow.start_run() as run:
        # Log params
        mlflow.log_params({
            "task": args.task,
            "model": args.model,
            "test_size": args.test_size,
            "random_state": args.random_state,
            "n_estimators": args.n_estimators if "n_estimators" in vars(args) else None,
            "max_depth": args.max_depth,
            "num_features": len(num_cols),
            "cat_features": len(cat_cols),
            "data_path": str(data_path),
        })

        pipe.fit(X_train, y_train)

        # Evaluate
        metrics = {}
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)

        if args.task == "classification":
            y_pred = pipe.predict(X_test)
            metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
            metrics["f1_macro"] = float(f1_score(y_test, y_pred, average="macro"))

            # AUC (jika binary dan ada predict_proba)
            try:
                if len(np.unique(y_test)) == 2 and hasattr(pipe, "predict_proba"):
                    y_proba = pipe.predict_proba(X_test)[:, 1]
                    metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
            except Exception:
                pass

            # Confusion matrix plot
            labels = sorted(pd.Series(y_test).unique().tolist())
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            cm_path = artifacts_dir / "confusion_matrix.png"
            plot_confusion_matrix(cm, labels, cm_path)
            mlflow.log_artifact(str(cm_path))

            # Classification report
            report = classification_report(y_test, y_pred, digits=4)
            (artifacts_dir / "classification_report.txt").write_text(report)
            mlflow.log_artifact(str(artifacts_dir / "classification_report.txt"))

        else:  # regression
            y_pred = pipe.predict(X_test)
            metrics["mae"] = float(mean_absolute_error(y_test, y_pred))
            metrics["rmse"] = float(mean_squared_error(y_test, y_pred, squared=False))
            metrics["r2"] = float(r2_score(y_test, y_pred))

        mlflow.log_metrics(metrics)

        # Save fitted column info as artifact (useful for serving)
        schema_info = {
            "numeric_features": num_cols,
            "categorical_features": cat_cols,
            "target": args.target,
            "task": args.task,
        }
        (artifacts_dir / "schema.json").write_text(json.dumps(schema_info, indent=2))
        mlflow.log_artifact(str(artifacts_dir / "schema.json"))

        # Log model (register kalau env diset & registry tersedia)
        reg_name = args.registered_model_name.strip() or None
        if reg_name:
            mlflow.sklearn.log_model(pipe, artifact_path="model", registered_model_name=reg_name)
        else:
            mlflow.sklearn.log_model(pipe, artifact_path="model")
        from mlflow.tracking import MlflowClient
        reg_name = (args.registered_model_name or os.getenv("MLFLOW_REGISTERED_MODEL","")).strip()
        if reg_name:
            client = MlflowClient()
            mv = next(int(m.version) for m in client.search_model_versions(f"name='{reg_name}'")
              if m.run_id == run.info.run_id)
            client.set_registered_model_alias(reg_name, "production", mv)
            print(f"Alias 'production' -> {reg_name} v{mv}")

        print("Run ID:", run.info.run_id)
        print("Metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
