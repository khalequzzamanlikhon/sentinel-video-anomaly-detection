FROM python:3.11-slim

# OpenCV needs these system libs even in headless mode.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY sentinel ./sentinel
COPY config ./config
COPY scripts ./scripts

RUN mkdir -p data/clips data/thumbnails

EXPOSE 8000 8501

CMD ["python", "-m", "sentinel.main", "--config", "config", "--headless"]
