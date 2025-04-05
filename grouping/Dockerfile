FROM python:3.11-slim as build

WORKDIR /app

# Install build tools and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

ARG NOCO_DB_URL
ARG NOCO_XC_TOKEN
ARG LLM_URL

ENV NOCO_DB_URL=${NOCO_DB_URL}
ENV NOCO_XC_TOKEN=${NOCO_XC_TOKEN}
ENV LLM_URL=${LLM_URL}

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5004

CMD ["python", "./app.py"]