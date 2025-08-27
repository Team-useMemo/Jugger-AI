from typing import List, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field, validator

from app.services.skt_text_processing import classify_paragraph

router = APIRouter()

class ParagraphRequest(BaseModel):
    paragraph: str = Field(..., description="분류할 전체 문단 텍스트")
    userCategories: Optional[List[str]] = Field(
        default=None, description="스프링에서 내려주는 카테고리명 리스트"
    )
    threshold: Optional[float] = Field(
        default=0.5, ge=0.0, le=1.0, description="카테고리 매칭 임계값(0~1)"
    )

    @validator("userCategories")
    def _strip_empty(cls, v):
        if v:
            v = [s.strip() for s in v if isinstance(s, str) and s.strip()]
            if len(v) == 0:
                return None
        return v

@router.post("/classify")
async def classify_paragraph_api(request: ParagraphRequest):
    result = await classify_paragraph(
        paragraph=request.paragraph,
        user_categories=request.userCategories,
        threshold=request.threshold or 0.5
    )
    return result
