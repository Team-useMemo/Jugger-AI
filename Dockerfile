FROM python:3.10-slim AS builder

WORKDIR /install

COPY requirements.txt ./

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install/packages -r requirements.txt

FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /install/packages /usr/local/

# 소스 전체 복사 (Dockerfile과 app 디렉토리가 같은 위치일 때)
COPY ./app /app

# FastAPI 인스턴스가 app.main 모듈 안에 있을 경우

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
