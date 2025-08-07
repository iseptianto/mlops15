# Instead of building from scratch, use pre-built images
# mlflow_server/Dockerfile
FROM python:3.9-slim

# Install system dependencies first
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Copy everything else
COPY . .

# Default command
CMD ["mlflow", "server", "--help"]