from fastapi import FastAPI
from app.routes import skt_classify
from app.routes import gemini
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = FastAPI(title="문장 분석 기능입니다.")

# API 라우터 등록
app.include_router(skt_classify.router, prefix="/ai", tags=["SKT KoBERT Classification"])
app.include_router(gemini.router, prefix="/ai", tags=["GEMINI Classification"])

@app.get("/")
def root():
    return {"message": "FastAPI SKT KoBERT 문장 분류 API (URL 추출 포함)"}
