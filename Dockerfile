FROM python:3.9-slim

WORKDIR /app

# Install system dependencies if any
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies list
COPY requirements-dev.txt /app/requirements.txt

# Install dependencies (CPU-only version of PyTorch to minimize image size)
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy project files
COPY ./api /app/api
COPY ./model /app/model
COPY ./public /app/public

# Expose port for Hugging Face Spaces
EXPOSE 7860

# Start FastAPI server using uvicorn
CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "7860"]
