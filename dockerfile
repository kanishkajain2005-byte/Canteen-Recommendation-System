FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000

ARG PORT=10000
ENV PORT=${PORT}

CMD ["sh", "-c", "uvicorn ML.API.api_general:app --host 0.0.0.0 --port ${PORT}"]
