FROM python:3.10-slim AS builder

WORKDIR /install
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install/packages -r requirements.txt

FROM python:3.10-slim

WORKDIR /app
ENV PYTHONPATH=/app

COPY --from=builder /install/packages /usr/local/
COPY ./app /app

CMD ["python", "server.py"]
