from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Текст письма для классификации (spam/ham)")


class PredictResponse(BaseModel):
    label: str = Field(..., description="Предсказанный класс: spam или ham")
    score: float = Field(..., ge=0.0, le=1.0, description="Уверенность модели для класса spam")
