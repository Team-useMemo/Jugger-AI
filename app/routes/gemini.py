from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.services.gemini_category import classify_paragraph_gemini

router = APIRouter()

class GeminiRequest(BaseModel):
    paragraph: str = Field(..., description="분류 대상 문단")
    userCategories: Optional[List[str]] = Field(default=None, description="카테고리 후보(선택)")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="카테고리 매칭 임계값(0~1)")
    k: int = Field(default=5, ge=1, le=10, description="추천 개수")

    @field_validator("userCategories")
    @classmethod
    def _strip_empty(cls, v):
        if v:
            v = [s.strip() for s in v if isinstance(s, str) and s.strip()]
            if not v:
                return None
        return v

class SentenceOut(BaseModel):
    text: str
    urls: Optional[List[str]] = None
    invalid_urls: Optional[List[str]] = None
    schedules: Optional[List[dict]] = None

class GeminiResponse(BaseModel):
    category: str
    recommend_category: List[str]
    sentences: List[SentenceOut]

@router.post("/gemini", response_model=GeminiResponse)
async def gemini_endpoint(req: GeminiRequest):
    try:
        return await classify_paragraph_gemini(
            paragraph=req.paragraph,
            user_categories=req.userCategories,
            threshold=req.threshold,
            k=req.k
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini 호출 실패: {e}")
