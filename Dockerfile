FROM python:3.10-slim AS builder

WORKDIR /install
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install/packages -r requirements.txt

FROM python:3.10-slim
WORKDIR /app
ENV PYTHONPATH=/app

# PYTHONPATH를 /app으로 설정 → 내부에서 모듈 import 문제 해결
ENV PYTHONPATH=/app

# 의존성 복사
COPY --from=builder /install/packages /usr/local/

# 코드 복사
COPY ./app /app

# server.py를 실행 (여기서 uvicorn.run이 직접 호출됨)
CMD ["python", "server.py"]
