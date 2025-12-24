# MLOps Spam Classifier (DistilBERT)

## Структура проекта
- `src/` — код обучения модели и препроцессинга данных  
- `app/` — сервис инференса на FastAPI  
- `configs/` — YAML-конфигурации проекта  
- `mlruns/` — эксперименты MLflow (игнорируется git)  
- `data/raw/` — сырой датасет (игнорируется git)  

## Установка и запуск
```bash
pip install -r requirements.txt