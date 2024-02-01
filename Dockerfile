FROM python:3.11-slim as build

WORKDIR /app

COPY . .

ARG NOCO_DB_URL
ARG NOCO_XC_TOKEN

ENV NOCO_DB_URL=${NOCO_DB_URL}
ENV NOCO_XC_TOKEN=${NOCO_XC_TOKEN}

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5001

CMD ["python", "./app.py"]