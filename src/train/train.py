import os
import yaml
import mlflow
import torch
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

class TextDataset(Dataset):
    """PyTorch dataset for text classification."""

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        
        text = str(self.texts[idx]) if self.texts[idx] is not None else ""

        enc = self.tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=self.max_length,
        return_tensors="pt",
        )       


        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_config(path="configs/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_split(version, split, config):
    base = Path(config["paths"]["processed_data"]) / version
    df = pd.read_csv(base / f"{split}.csv")
    return df


def evaluate(model, dataloader, device):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1)

            preds.extend(pred.cpu().numpy())
            targets.extend(batch["labels"].cpu().numpy())

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds)

    return acc, f1


def main():

    config = load_config()

    print("=== TRAIN START ===")
    print(f"Dataset version: {config['data'].get('dataset_version', 'v1')}")
    print(f"Device config: {config['train'].get('device', 'auto')}")

    # --- MLflow ---
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # --- Config params ---
    version = config["data"]["dataset_version"]
    model_name = config["train"]["base_model_name"]
    lr = float(config["train"]["lr"])
    epochs = int(config["train"]["epochs"])
    batch_size = int(config["train"]["batch_size"])
    max_length = int(config["train"]["max_length"])
    weight_decay = float(config["train"]["weight_decay"])


    device = "cuda" if torch.cuda.is_available() else "cpu"

    with mlflow.start_run(run_name=config["mlflow"]["run_name"]):
        # Log params
        mlflow.log_param("dataset_version", version)
        mlflow.log_param("model", model_name)
        mlflow.log_param("lr", lr)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("max_length", max_length)
        mlflow.log_param("weight_decay", weight_decay)


        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        ).to(device)

        # Load data
        df_train = load_split(version, "train", config)
        df_val = load_split(version, "val", config)

        train_ds = TextDataset(
            df_train["text"].tolist(),
            df_train["label"].tolist(),
            tokenizer,
            max_length,
        )
        val_ds = TextDataset(
            df_val["text"].tolist(),
            df_val["label"].tolist(),
            tokenizer,
            max_length,
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # --- Training loop ---
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0

            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            val_acc, val_f1 = evaluate(model, val_loader, device)

            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            mlflow.log_metric("val_f1", val_f1, step=epoch)

            print(
                f"Epoch {epoch}: loss={avg_loss:.4f}, "
                f"val_acc={val_acc:.4f}, val_f1={val_f1:.4f}"
            )

        # Save model
        model_dir = Path(config["paths"]["model_dir"]) / f"run_{mlflow.active_run().info.run_id}"
        model_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

        mlflow.log_artifacts(str(model_dir), artifact_path="model")


if __name__ == "__main__":
    main()
