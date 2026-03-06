import json
import logging
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─── Konfigurasi ──────────────────────────────────────────────────────────────
PREPROCESSING_DIR = os.getenv("PREPROCESSING_DIR", "twitter_emotion_preprocessing")
RANDOM_SEED       = int(os.getenv("RANDOM_SEED", "42"))
EXPERIMENT_NAME   = "Indonesian-Emotion-Classification"
ARTIFACTS_DIR     = "./artifacts_modelling"

np.random.seed(RANDOM_SEED)


def run_training():
    """Isi training — dipanggil di dalam run yang sudah aktif."""

    run = mlflow.active_run()
    log.info("MLflow Run ID: %s", run.info.run_id)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────────
    log.info("Memuat dataset dari: %s", PREPROCESSING_DIR)
    df_train = pd.read_csv(os.path.join(PREPROCESSING_DIR, "train.csv"))
    df_val   = pd.read_csv(os.path.join(PREPROCESSING_DIR, "val.csv"))

    with open(os.path.join(PREPROCESSING_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)

    class_names = list(le.classes_)
    log.info("Train: %d | Val: %d | Labels: %s", len(df_train), len(df_val), class_names)

    X_train = df_train["clean_tweet"].astype(str).tolist()
    y_train = df_train["label_id"].tolist()
    X_val   = df_val["clean_tweet"].astype(str).tolist()
    y_val   = df_val["label_id"].tolist()

    # ── Pipeline ──────────────────────────────────────────────────────────────
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )),
    ])

    # Autolog aktif tapi log_models=False (kita log manual)
    mlflow.sklearn.autolog(log_models=False)

    # ── Log params ────────────────────────────────────────────────────────────
    mlflow.log_params({
        "model_type"    : "RandomForest",
        "n_estimators"  : 100,
        "max_features"  : 10000,
        "ngram_range"   : "(1,2)",
        "random_seed"   : RANDOM_SEED,
        "train_samples" : len(X_train),
        "val_samples"   : len(X_val),
        "num_labels"    : len(class_names),
    })

    # ── Training ──────────────────────────────────────────────────────────────
    log.info("Training model...")
    pipeline.fit(X_train, y_train)

    # ── Evaluasi ──────────────────────────────────────────────────────────────
    y_pred = pipeline.predict(X_val)

    acc  = accuracy_score(y_val, y_pred)
    f1w  = f1_score(y_val, y_pred, average="weighted")
    f1m  = f1_score(y_val, y_pred, average="macro")
    prec = precision_score(y_val, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_val, y_pred, average="weighted", zero_division=0)

    mlflow.log_metrics({
        "accuracy"   : acc,
        "f1_weighted": f1w,
        "f1_macro"   : f1m,
        "precision"  : prec,
        "recall"     : rec,
    })

    log.info("Accuracy   : %.4f", acc)
    log.info("F1 Weighted: %.4f", f1w)

    # ── Log model ─────────────────────────────────────────────────────────────
    mlflow.sklearn.log_model(
        sk_model      = pipeline,
        artifact_path = "model",
    )
    log.info("Model logged!")

    # ── Artefak 1: Confusion Matrix ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(
        confusion_matrix(y_val, y_pred),
        display_labels=class_names,
    ).plot(ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix — Random Forest", fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cm_path = os.path.join(ARTIFACTS_DIR, "training_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(cm_path)
    log.info("Confusion matrix logged!")

    # ── Artefak 2: Classification Report JSON ─────────────────────────────────
    report = classification_report(
        y_val, y_pred,
        target_names=class_names,
        output_dict=True,
        digits=4,
    )
    cr_path = os.path.join(ARTIFACTS_DIR, "classification_report.json")
    with open(cr_path, "w") as f:
        json.dump(report, f, indent=2)
    mlflow.log_artifact(cr_path)
    log.info("Classification report logged!")

    # ── Simpan run_id untuk workflow CI ───────────────────────────────────────
    with open("latest_run_id.txt", "w") as f:
        f.write(run.info.run_id)

    log.info("=" * 60)
    log.info("SELESAI — Run ID: %s", run.info.run_id)
    log.info("=" * 60)


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)
    log.info("Experiment : %s", EXPERIMENT_NAME)

    # Cek apakah sudah ada active run (dijalankan via `mlflow run`)
    # atau perlu buat sendiri (dijalankan manual via `python modelling.py`)
    if mlflow.active_run() is not None:
        # Sudah ada run dari MLflow Project — langsung pakai
        log.info("Menggunakan active run dari MLflow Project")
        run_training()
    else:
        # Dijalankan manual — buat run sendiri
        log.info("Membuat run baru (mode manual)")
        with mlflow.start_run(run_name="rf-tfidf-autolog"):
            run_training()


if __name__ == "__main__":
    main()