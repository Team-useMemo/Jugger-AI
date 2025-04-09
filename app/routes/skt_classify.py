from fastapi import APIRouter
from pydantic import BaseModel
from app.services.skt_text_processing import classify_paragraph_with_user

router = APIRouter()

class ParagraphRequest(BaseModel):
    user_uuid: str
    paragraph: str

@router.post("/ai/classify")
async def classify_paragraph_api(request: ParagraphRequest):
    result = await classify_paragraph_with_user(request.user_uuid, request.paragraph)
    return result
