# Use official Python slim image
FROM python:3.10-slim

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all project files
COPY . /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Optional: download models (skipped if models are already there or download fails)
RUN python setup/download_models.py || echo "⚠️ Model download skipped or failed."

# Expose Streamlit's default port
EXPOSE 8501

# Start the Streamlit UI
CMD ["streamlit", "run", "inference/streamlit.py", "--server.port=8501", "--server.enableCORS=false"]
