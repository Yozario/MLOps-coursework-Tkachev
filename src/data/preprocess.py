import re
import os
import yaml
import pandas as pd
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Подгрузка YAML файл конфигурации."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_text_v1(text: str) -> str:
    """базовая чистка текста(v1)."""
    text = str(text)
    text = text.strip()
    return text


def clean_text_v2(text: str) -> str:
    """Расширенная чистка текста (v2)."""
    text = str(text)
    text = text.lower()
    text = text.strip()

    # Убираем URLы
    text = re.sub(r"http\S+|www\S+", "", text)

    # Удаляем HTML теги
    text = re.sub(r"<.*?>", "", text)

    return text


def preprocess_dataset(config_path: str):
    """Основной пайплайн предобработки."""
    config = load_config(config_path)

    raw_path = Path(config["paths"]["raw_data"])
    processed_path = Path(config["paths"]["processed_data"])
    raw_file = raw_path / config["data"]["raw_filename"]

    if not raw_file.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_file}")

    df = pd.read_csv(raw_file)

    text_col = config["data"]["text_col"]
    label_col = config["data"]["label_col"]

    # Нормализуем лейблы: Spam -> 1, Ham -> 0
    df[label_col] = df[label_col].map({"Spam": 1, "Ham": 0})

    # -------- v1 --------
    v1_dir = processed_path / "v1"
    v1_dir.mkdir(parents=True, exist_ok=True)

    df_v1 = df.copy()
    df_v1[text_col] = df_v1[text_col].apply(clean_text_v1)

    df_v1.to_csv(v1_dir / "data.csv", index=False)

    # -------- v2 --------
    v2_dir = processed_path / "v2"
    v2_dir.mkdir(parents=True, exist_ok=True)

    df_v2 = df.copy()
    df_v2[text_col] = df_v2[text_col].apply(clean_text_v2)

    df_v2.to_csv(v2_dir / "data.csv", index=False)

    print("Предобработка прошла успешно.")
    print(f"v1 датасет сохранён в : {v1_dir}")
    print(f"v2 датасет сохранён в : {v2_dir}")


if __name__ == "__main__":
    preprocess_dataset("configs/config.yaml")
