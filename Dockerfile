FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    COQUI_TOS_AGREED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    espeak-ng \
    libsndfile1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu \
      torch==2.5.1 \
      torchaudio==2.5.1 && \
    pip install \
      coqui-tts \
      fastapi==0.115.0 \
      "uvicorn[standard]==0.32.0" \
      numpy==1.26.4

COPY preload_models.py main.py ./

RUN python preload_models.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
