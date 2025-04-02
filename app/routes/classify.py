from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from services.text_processing import classify_or_create_category, category_embeddings

router = APIRouter()

# 요청 데이터 형식 정의
class SentenceRequest(BaseModel):
    sentences: List[str]  # 문자열 리스트 받기

# 문자열 입력 후 문장 자동 분류 API
@router.post("/classify")
async def classify_text(sentences: SentenceRequest):
    categorized_results = classify_or_create_category(sentences.sentences)
    return {"categorized_sentences": categorized_results, "category_list": list(category_embeddings.keys())}
