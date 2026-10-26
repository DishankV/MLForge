
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --no-cache-dir -r requirements.txt

COPY inference/ ./inference/
COPY ml/registry/models/churn-logreg-v1/ ./ml/registry/models/churn-logreg-v1/

EXPOSE 5000

CMD ["python", "inference/app.py"]
