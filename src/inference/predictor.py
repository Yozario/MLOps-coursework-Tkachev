from __future__ import annotations

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def resolve_device(device_cfg: str) -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class SpamPredictor:
    """Класс-обёртка для инференса модели."""

    def __init__(self, model_path: str, max_length: int = 128, device: str = "auto", threshold: float = 0.5):
        self.model_path = model_path
        self.max_length = int(max_length)
        self.threshold = float(threshold)
        self.device = resolve_device(device)

        # Загружаем tokenizer и модель из папки models/best
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, text: str) -> tuple[str, float]:
        """Возвращает (label, score_spam)."""
        if not isinstance(text, str):
            text = str(text)

        enc = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        logits = self.model(**enc).logits

        # В бинарной классификации обычно num_labels=2: [ham, spam]
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        score_spam = float(probs[1].item())

        label = "spam" if score_spam >= self.threshold else "ham"
        return label, score_spam
