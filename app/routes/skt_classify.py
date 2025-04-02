from fastapi import APIRouter
from pydantic import BaseModel
from services.skt_text_processing import classify_paragraph

router = APIRouter()

# 요청 데이터 형식 정의
class ParagraphRequest(BaseModel):
    paragraph: str  # 단일 문단 받기

# ✅ 문단을 입력 후 카테고리 분류 API
@router.post("/classify_paragraph")
async def classify_paragraph_api(request: ParagraphRequest):
    result = classify_paragraph(request.paragraph)
    return result
