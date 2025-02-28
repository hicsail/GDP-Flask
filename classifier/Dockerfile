FROM python:3.11-slim as build

WORKDIR /app

ARG NOCO_DB_URL
ARG NOCO_XC_TOKEN
ARG LLM_URL

ENV NOCO_DB_URL=${NOCO_DB_URL}
ENV NOCO_XC_TOKEN=${NOCO_XC_TOKEN}
ENV LLM_URL=${LLM_URL}

ENV PYTHONUNBUFFERED=1

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libnss3 \
    libnspr4 \
    libdbus-1-3 \
    libatk1.0-0 \
    libexpat1 \
    libatspi2.0-0 \
    libx11-6 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libxcb1 \
    libxkbcommon0 \
    libasound2 \
    ca-certificates \
    wget \
    fonts-liberation \
    libappindicator3-1 \
    lsb-release \
    xdg-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN playwright install

COPY . .

EXPOSE 5001

CMD ["python", "./app.py"]