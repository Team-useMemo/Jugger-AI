FROM python:3.10-slim AS builder

WORKDIR /install

COPY requirements.txt ./

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install/packages -r requirements.txt

FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /install/packages /usr/local/

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
