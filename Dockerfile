 FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
 "gunicorn --worker-class gthread --threads 4 --timeout 0 --bind 0.0.0.0:${PORT:-8000} app:app"]