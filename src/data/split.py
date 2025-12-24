import os
import yaml
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_config(config_path: str) -> dict:
    """Загружаем YAML-конфиг."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_splits(df: pd.DataFrame, label_col: str, seed: int, train_size: float, val_size: float, test_size: float):
    """Делаем разбиение train/val/test со стратификацией по метке."""
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Сумма долей должна быть 1.0"

    df_train, df_tmp = train_test_split(
        df,
        test_size=(1.0 - train_size),
        random_state=seed,
        stratify=df[label_col],
    )

    # val и test делим из временной части
    rel_test = test_size / (val_size + test_size)
    df_val, df_test = train_test_split(
        df_tmp,
        test_size=rel_test,
        random_state=seed,
        stratify=df_tmp[label_col],
    )

    return df_train, df_val, df_test


def split_for_version(version: str, config: dict):
    """Создаём сплиты для указанной версии датасета (v1 или v2)."""
    processed_dir = Path(config["paths"]["processed_data"]) / version
    input_file = processed_dir / "data.csv"

    if not input_file.exists():
        raise FileNotFoundError(f"Не найден файл датасета: {input_file}. Сначала запусти preprocess.py")

    df = pd.read_csv(input_file)
    label_col = config["data"]["label_col"]

    seed = int(config["project"]["seed"])
    split_cfg = config["data"]["split"]
    train_size = float(split_cfg["train_size"])
    val_size = float(split_cfg["val_size"])
    test_size = float(split_cfg["test_size"])

    df_train, df_val, df_test = make_splits(df, label_col, seed, train_size, val_size, test_size)

    df_train.to_csv(processed_dir / "train.csv", index=False)
    df_val.to_csv(processed_dir / "val.csv", index=False)
    df_test.to_csv(processed_dir / "test.csv", index=False)

    print(f"[{version}] splits saved:")
    print(f"  train: {len(df_train)} rows")
    print(f"  val:   {len(df_val)} rows")
    print(f"  test:  {len(df_test)} rows")


def main():
    config = load_config("configs/config.yaml")

    # Делаем сплиты для обеих версий, чтобы дальше легко гонять эксперименты
    for version in ["v1", "v2"]:
        split_for_version(version, config)


if __name__ == "__main__":
    main()
