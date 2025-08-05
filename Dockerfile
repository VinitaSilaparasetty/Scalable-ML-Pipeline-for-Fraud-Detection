# Dockerfile for scalable-ml-pipeline
FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["python", "main.py"]
