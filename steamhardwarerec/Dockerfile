# === Dockerfile for Steam Hardware Recommender ===
# Base image: Official Python slim for smaller size
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for potential build issues)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (leverages Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Health check (optional but recommended)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit app
CMD ["streamlit", "run", "predict.py", "--server.port=8501", "--server.address=0.0.0.0"]