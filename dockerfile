# Stage 1: Build Stage (Uses a more feature-rich image for package compilation if needed)
# Using a 'slim' base image is recommended for Python as it's much smaller than the default
FROM python:3.11-slim as base

# Set the working directory for all subsequent instructions
WORKDIR /app

# Copy the dependency file first to take advantage of Docker layer caching.
# If requirements.txt doesn't change, this and the next step are skipped on rebuilds.
COPY requirements.txt requirements.txt

# Install Python dependencies.
# --no-cache-dir reduces image size by not storing the pip cache.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code and trained model file(s)
# The .dockerignore file should exclude unnecessary files like .git, venv, etc.
COPY . .

# EXPOSE the port your application will run on (e.g., 8000 for a FastAPI/Flask API)
# This is documentation and doesn't actually publish the port.
EXPOSE 8000

# Specify the command to run the application when the container starts.
# Use the exec form (array syntax) for better performance and signal handling.
# Replace 'app.py' with your primary script name.
CMD ["python", "recommend_for_user.py"]