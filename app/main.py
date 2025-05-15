from fastapi import FastAPI
from routes import skt_classify

app = FastAPI(title="SKT KoBERT Text Classification API with URL Extraction")

# API 라우터 등록
app.include_router(skt_classify.router, prefix="/api", tags=["SKT KoBERT Classification"])

@app.get("/")
def root():
    return {"message": "FastAPI SKT KoBERT 문장 분류 API (URL 추출 포함)"}
