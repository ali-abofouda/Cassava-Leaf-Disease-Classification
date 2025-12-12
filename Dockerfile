# Use Python 3.9 slim base image for smaller size
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker caching
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch CPU version (much smaller than CUDA version)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy application files
COPY . .

# Grant read/write permissions to avoid permission issues on Hugging Face Spaces
RUN chmod -R 777 /app

# Expose port 7860 (required by Hugging Face Spaces)
EXPOSE 7860

# Run the application with gunicorn
# - bind to 0.0.0.0:7860 for external access
# - 2 workers for handling concurrent requests
# - timeout of 120 seconds for model inference
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "120", "app:app"]
