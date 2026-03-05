import json
import logging
import os
import pickle
import sys

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    EarlyStoppingCallback, Trainer, TrainingArguments,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

PREPROCESSING_DIR = os.getenv("PREPROCESSING_DIR", "twitter_emotion_preprocessing")
MODEL_NAME        = os.getenv("MODEL_NAME",         "indobenchmark/indobert-base-p1")
MAX_LENGTH        = int(os.getenv("MAX_LENGTH",     "128"))
BATCH_SIZE        = int(os.getenv("BATCH_SIZE",     "16"))
NUM_EPOCHS        = int(os.getenv("NUM_EPOCHS",     "1"))
LEARNING_RATE     = float(os.getenv("LEARNING_RATE","2e-5"))
WEIGHT_DECAY      = float(os.getenv("WEIGHT_DECAY", "0.05"))
WARMUP_RATIO      = float(os.getenv("WARMUP_RATIO", "0.1"))
RANDOM_SEED       = int(os.getenv("RANDOM_SEED",    "42"))
CI_MODE           = os.getenv("CI_MODE", "false").lower() == "true"
CI_SAMPLE_SIZE    = int(os.getenv("CI_SAMPLE_SIZE", "300"))
OUTPUT_DIR        = os.getenv("OUTPUT_DIR",         "./results_ci")
EXPERIMENT_NAME   = "Indonesian-Emotion-CI"

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    return {
        "accuracy"   : accuracy_score(labels, preds),
        "f1_macro"   : f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "precision"  : precision_score(labels, preds, average="weighted", zero_division=0),
        "recall"     : recall_score(labels, preds, average="weighted", zero_division=0),
    }


def main():
    log.info("=" * 60)
    log.info("MLflow Project — IndoBERT CI Training")
    log.info("CI Mode: %s | Epochs: %d", CI_MODE, NUM_EPOCHS)
    log.info("=" * 60)

    # Setup MLflow lokal
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load dataset
    train_path = os.path.join(PREPROCESSING_DIR, "train.csv")
    val_path   = os.path.join(PREPROCESSING_DIR, "val.csv")

    if not os.path.exists(train_path):
        log.error("Dataset tidak ditemukan: %s", train_path)
        sys.exit(1)

    df_train = pd.read_csv(train_path)
    df_val   = pd.read_csv(val_path)

    if CI_MODE:
        log.info("CI Mode — sample %d data", CI_SAMPLE_SIZE)
        n = max(1, CI_SAMPLE_SIZE // df_train["label"].nunique())
        df_train = (
            df_train.groupby("label", group_keys=False)
            .apply(lambda x: x.sample(min(len(x), n), random_state=RANDOM_SEED),
                   include_groups=False)
            .reset_index(drop=True)
        )
        df_val = df_val.sample(
            min(len(df_val), CI_SAMPLE_SIZE // 4), random_state=RANDOM_SEED
        ).reset_index(drop=True)
        log.info("Setelah sampling — train: %d, val: %d", len(df_train), len(df_val))

    with open(os.path.join(PREPROCESSING_DIR, "metadata.json")) as f:
        metadata = json.load(f)
    with open(os.path.join(PREPROCESSING_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)

    num_labels = metadata["num_labels"]
    log.info("Train: %d | Val: %d | Labels: %d", len(df_train), len(df_val), num_labels)

    # Model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels,
        problem_type="single_label_classification",
    )
    model.config.hidden_dropout_prob          = 0.3
    model.config.attention_probs_dropout_prob = 0.3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    log.info("Device: %s", device)

    train_dataset = EmotionDataset(
        df_train["clean_tweet"].tolist(), df_train["label_id"].tolist(),
        tokenizer, MAX_LENGTH,
    )
    val_dataset = EmotionDataset(
        df_val["clean_tweet"].tolist(), df_val["label_id"].tolist(),
        tokenizer, MAX_LENGTH,
    )

    training_args = TrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = NUM_EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        warmup_ratio                = WARMUP_RATIO,
        weight_decay                = WEIGHT_DECAY,
        learning_rate               = LEARNING_RATE,
        logging_steps               = 10,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1_weighted",
        greater_is_better           = True,
        save_total_limit            = 1,
        report_to                   = "none",
        fp16                        = torch.cuda.is_available(),
        seed                        = RANDOM_SEED,
    )

    with mlflow.start_run() as run:
        log.info("MLflow Run ID: %s", run.info.run_id)

        # Log params — semua string/int agar tidak error
        mlflow.log_param("model_name",    str(MODEL_NAME))
        mlflow.log_param("max_length",    int(MAX_LENGTH))
        mlflow.log_param("batch_size",    int(BATCH_SIZE))
        mlflow.log_param("num_epochs",    int(NUM_EPOCHS))
        mlflow.log_param("learning_rate", "2e-5")
        mlflow.log_param("weight_decay",  "0.05")
        mlflow.log_param("warmup_ratio",  "0.1")
        mlflow.log_param("train_samples", int(len(train_dataset)))
        mlflow.log_param("val_samples",   int(len(val_dataset)))
        mlflow.log_param("num_labels",    int(num_labels))
        mlflow.log_param("ci_mode",       str(CI_MODE))
        mlflow.log_param("device",        str(device))

        trainer = Trainer(
            model=model, args=training_args,
            train_dataset=train_dataset, eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=2, early_stopping_threshold=0.001,
            )],
        )

        log.info("Mulai training...")
        trainer.train()

        eval_results = trainer.evaluate()
        mlflow.log_metric("eval_accuracy",    eval_results.get("eval_accuracy", 0))
        mlflow.log_metric("eval_f1_macro",    eval_results.get("eval_f1_macro", 0))
        mlflow.log_metric("eval_f1_weighted", eval_results.get("eval_f1_weighted", 0))
        mlflow.log_metric("eval_precision",   eval_results.get("eval_precision", 0))
        mlflow.log_metric("eval_recall",      eval_results.get("eval_recall", 0))
        mlflow.log_metric("eval_loss",        eval_results.get("eval_loss", 0))

        # Artefak: confusion matrix
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        predictions = trainer.predict(val_dataset)
        y_pred = predictions.predictions.argmax(-1)
        y_true = predictions.label_ids

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
        fig, ax = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay(
            confusion_matrix(y_true, y_pred), display_labels=le.classes_,
        ).plot(ax=ax, cmap="Blues")
        ax.set_title("Confusion Matrix — CI", fontweight="bold")
        plt.tight_layout()
        plt.savefig(cm_path, dpi=120, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # Artefak: classification report
        report = classification_report(y_true, y_pred,
            target_names=le.classes_, output_dict=True)
        report_path = os.path.join(OUTPUT_DIR, "classification_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path, artifact_path="reports")

        # Simpan model config (bukan bobot besar)
        model_config_path = os.path.join(OUTPUT_DIR, "model_config")
        os.makedirs(model_config_path, exist_ok=True)
        trainer.model.config.to_json_file(
            os.path.join(model_config_path, "config.json"))
        tokenizer.save_pretrained(model_config_path)
        mlflow.log_artifacts(model_config_path, artifact_path="model_config")

        # Simpan run_id
        with open("latest_run_id.txt", "w") as f:
            f.write(run.info.run_id)

        log.info("=" * 60)
        log.info("✅ SELESAI | Run ID: %s", run.info.run_id)
        log.info("   Accuracy    : %.4f", eval_results.get("eval_accuracy", 0))
        log.info("   F1 Weighted : %.4f", eval_results.get("eval_f1_weighted", 0))
        log.info("=" * 60)


if __name__ == "__main__":
    main()