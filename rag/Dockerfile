FROM python:3.11-slim as build

WORKDIR /app

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

EXPOSE 5005

CMD ["python", "./app.py"]