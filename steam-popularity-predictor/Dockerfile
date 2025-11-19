FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (code, data, models dir)
COPY . .

# Pre-train the model during build (assumes data in archive/)
RUN python train.py

# Create models dir if needed (train.py does this)
RUN mkdir -p models

# Expose ports
EXPOSE 8000 8501

# Default CMD: Run FastAPI (override for Streamlit via docker run -e CMD="streamlit run ...")
CMD ["uvicorn", "app2:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]