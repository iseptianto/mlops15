FROM python:3.9-slim

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (if exists)
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Install common ML packages
RUN pip install --no-cache-dir \
    mlflow==2.8.1 \
    boto3==1.29.7 \
    psycopg2-binary==2.9.9 \
    pandas==1.5.3 \
    scikit-learn==1.3.2 \
    numpy==1.24.3

# Copy all project files
COPY . .

# Default command (akan di-override oleh docker-compose)
CMD ["python", "generate_dummy_pickle.py"]