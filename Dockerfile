FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip install --no-cache-dir \
        fastapi \
        uvicorn \
        mlflow \
        dagshub \
        dvc \
        dvc-http \
        pandas \
        numpy \
        scikit-learn \
        joblib \
        xgboost \
        holidays \
        python-dotenv \
        requests \
        beautifulsoup4 \
        lxml

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]