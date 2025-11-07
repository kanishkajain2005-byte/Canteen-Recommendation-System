# 1. Use an official Python image
FROM python:3.10-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy project files into the container
COPY . /app

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Expose the port that your FastAPI runs on
EXPOSE 8000

# 6. Start the server
CMD ["uvicorn", "ML.API.recommend_api:app", "--host", "0.0.0.0", "--port", "8000"]
