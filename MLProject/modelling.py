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
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Konfigurasi dari environment variable / default 
PREPROCESSING_DIR   = os.getenv("PREPROCESSING_DIR",   "twitter_emotion_preprocessing")
MODEL_NAME          = os.getenv("MODEL_NAME",           "indobenchmark/indobert-base-p1")
MAX_LENGTH          = int(os.getenv("MAX_LENGTH",       "128"))
BATCH_SIZE          = int(os.getenv("BATCH_SIZE",       "16"))
NUM_EPOCHS          = int(os.getenv("NUM_EPOCHS",       "3"))
LEARNING_RATE       = float(os.getenv("LEARNING_RATE",  "2e-5"))
WEIGHT_DECAY        = float(os.getenv("WEIGHT_DECAY",   "0.05"))
WARMUP_RATIO        = float(os.getenv("WARMUP_RATIO",   "0.1"))
RANDOM_SEED         = int(os.getenv("RANDOM_SEED",      "42"))
# CI_MODE: jika True, pakai sample kecil agar cepat di GitHub Actions
CI_MODE             = os.getenv("CI_MODE", "false").lower() == "true"
CI_SAMPLE_SIZE      = int(os.getenv("CI_SAMPLE_SIZE",  "500"))
OUTPUT_DIR          = os.getenv("OUTPUT_DIR",           "./results_ci")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# DATASET CLASS
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids"     : encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels"        : torch.tensor(self.labels[idx], dtype=torch.long),
        }

# METRICS
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

# MAIN
def main():
    log.info("=" * 60)
    log.info("MLFLOW PROJECT — IndoBERT CI Training")
    log.info("=" * 60)
    log.info("CI Mode      : %s", CI_MODE)
    log.info("Sample Size  : %s", CI_SAMPLE_SIZE if CI_MODE else "Full dataset")
    log.info("Epochs       : %d", NUM_EPOCHS)
    log.info("Learning Rate: %g", LEARNING_RATE)
    log.info("=" * 60)

    # 1. Load dataset
    train_path = os.path.join(PREPROCESSING_DIR, "train.csv")
    val_path   = os.path.join(PREPROCESSING_DIR, "val.csv")
    meta_path  = os.path.join(PREPROCESSING_DIR, "metadata.json")
    le_path    = os.path.join(PREPROCESSING_DIR, "label_encoder.pkl")

    if not os.path.exists(train_path):
        log.error("Dataset tidak ditemukan: %s", train_path)
        sys.exit(1)

    df_train = pd.read_csv(train_path)
    df_val   = pd.read_csv(val_path)

    # Jika CI mode, ambil sample kecil per kelas agar training cepat
    if CI_MODE:
        log.info("CI Mode aktif — menggunakan sample %d data", CI_SAMPLE_SIZE)
        n_per_class = max(1, CI_SAMPLE_SIZE // df_train["label"].nunique())
        df_train = (
            df_train.groupby("label", group_keys=False)
                    .apply(lambda x: x.sample(min(len(x), n_per_class), random_state=RANDOM_SEED))
                    .reset_index(drop=True)
        )
        df_val = df_val.sample(
            min(len(df_val), CI_SAMPLE_SIZE // 4),
            random_state=RANDOM_SEED,
        ).reset_index(drop=True)
        log.info("Setelah sampling — train: %d, val: %d", len(df_train), len(df_val))

    with open(meta_path) as f:
        metadata = json.load(f)
    with open(le_path, "rb") as f:
        le = pickle.load(f)

    num_labels = metadata["num_labels"]
    log.info("Train: %d | Val: %d | Labels: %d", len(df_train), len(df_val), num_labels)

    # 2. Load tokenizer & model
    log.info("Memuat tokenizer: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    log.info("Memuat model IndoBERT...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        problem_type="single_label_classification",
    )
    model.config.hidden_dropout_prob          = 0.3
    model.config.attention_probs_dropout_prob = 0.3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    log.info("Device: %s", device)

    # 3. Dataset objects
    train_dataset = EmotionDataset(
        df_train["clean_tweet"].tolist(),
        df_train["label_id"].tolist(),
        tokenizer, MAX_LENGTH,
    )
    val_dataset = EmotionDataset(
        df_val["clean_tweet"].tolist(),
        df_val["label_id"].tolist(),
        tokenizer, MAX_LENGTH,
    )

    # 4. Training arguments
    training_args = TrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = NUM_EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        warmup_ratio                = WARMUP_RATIO,
        weight_decay                = WEIGHT_DECAY,
        learning_rate               = LEARNING_RATE,
        logging_dir                 = os.path.join(OUTPUT_DIR, "logs"),
        logging_steps               = 20,
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

    # 5. MLflow manual logging
    with mlflow.start_run() as run:
        log.info("MLflow Run ID: %s", run.info.run_id)

        # Log parameters
        mlflow.log_params({
            "model_name"    : MODEL_NAME,
            "max_length"    : MAX_LENGTH,
            "batch_size"    : BATCH_SIZE,
            "num_epochs"    : NUM_EPOCHS,
            "learning_rate" : LEARNING_RATE,
            "weight_decay"  : WEIGHT_DECAY,
            "warmup_ratio"  : WARMUP_RATIO,
            "train_samples" : len(train_dataset),
            "val_samples"   : len(val_dataset),
            "num_labels"    : num_labels,
            "ci_mode"       : str(CI_MODE),
            "device"        : str(device),
        })

        # 6. Training
        trainer = Trainer(
            model           = model,
            args            = training_args,
            train_dataset   = train_dataset,
            eval_dataset    = val_dataset,
            compute_metrics = compute_metrics,
            callbacks       = [EarlyStoppingCallback(
                early_stopping_patience  = 2,
                early_stopping_threshold = 0.001,
            )],
        )

        log.info("Memulai training...")
        trainer.train()

        # 7. Evaluasi & log metrics
        eval_results = trainer.evaluate()

        mlflow.log_metrics({
            "eval_accuracy"   : eval_results.get("eval_accuracy", 0),
            "eval_f1_macro"   : eval_results.get("eval_f1_macro", 0),
            "eval_f1_weighted": eval_results.get("eval_f1_weighted", 0),
            "eval_precision"  : eval_results.get("eval_precision", 0),
            "eval_recall"     : eval_results.get("eval_recall", 0),
            "eval_loss"       : eval_results.get("eval_loss", 0),
        })

        # 8. Artefak: confusion matrix
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        predictions = trainer.predict(val_dataset)
        y_pred      = predictions.predictions.argmax(-1)
        y_true      = predictions.label_ids

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
        fig, ax = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay(
            confusion_matrix(y_true, y_pred),
            display_labels=le.classes_,
        ).plot(ax=ax, cmap="Blues")
        ax.set_title("Confusion Matrix — CI Training", fontweight="bold")
        plt.tight_layout()
        plt.savefig(cm_path, dpi=120, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # 9. Artefak: classification report
        report = classification_report(
            y_true, y_pred,
            target_names=le.classes_,
            output_dict=True,
        )
        report_path = os.path.join(OUTPUT_DIR, "classification_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path, artifact_path="reports")

        # 10. Log model
        mlflow.pytorch.log_model(model, artifact_path="model")

        # 11. Simpan run_id untuk diambil di workflow
        run_id_path = "latest_run_id.txt"
        with open(run_id_path, "w") as f:
            f.write(run.info.run_id)
        mlflow.log_artifact(run_id_path)

        log.info("=" * 60)
        log.info("✅ TRAINING SELESAI")
        log.info("   Run ID      : %s", run.info.run_id)
        log.info("   Accuracy    : %.4f", eval_results.get("eval_accuracy", 0))
        log.info("   F1 Weighted : %.4f", eval_results.get("eval_f1_weighted", 0))
        log.info("=" * 60)

        return run.info.run_id


if __name__ == "__main__":
    main()