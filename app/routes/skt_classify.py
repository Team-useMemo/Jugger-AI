from fastapi import APIRouter
from pydantic import BaseModel
from app.services.skt_text_processing import classify_paragraph

router = APIRouter()

class ParagraphRequest(BaseModel):
    paragraph: str  # 단일 문단 받기

@router.post("/classify_paragraph")
async def classify_paragraph_api(request: ParagraphRequest):
    result = classify_paragraph(request.paragraph)
    return result
