from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DeviceType = Literal["cpu", "cuda", "auto"]


@dataclass
class PredictorConfig:
    # Путь к папке модели (transformers format: config.json, model.safetensors, tokenizer files)
    model_path: str = "models/best"
    # Максимальная длина последовательности для токенизатора
    max_length: int = 128
    # Устройство: auto/cpu/cuda
    device: DeviceType = "auto"
    # Порог (на будущее, если будешь выдавать probability)
    threshold: float = 0.5


class SpamPredictor:
    def __init__(self, cfg: PredictorConfig):
        self.cfg = cfg
        self.device = self._resolve_device(cfg.device)

        model_dir = Path(cfg.model_path)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        # Загружаем tokenizer + model из локальной папки (models/best)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir.as_posix())
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir.as_posix())
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device(device: DeviceType) -> torch.device:
        if device == "cpu":
            return torch.device("cpu")
        if device == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # auto
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.inference_mode()
    def predict(self, text: str) -> Tuple[str, float]:
        # Возвращаем (label, score)
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logits = self.model(**inputs).logits  # shape [1,2]
        probs = torch.softmax(logits, dim=-1).squeeze(0)

        ham_prob = float(probs[0].item())
        spam_prob = float(probs[1].item())

        label = "spam" if spam_prob >= 0.5 else "ham"
        score = spam_prob if label == "spam" else ham_prob
        return label, score
