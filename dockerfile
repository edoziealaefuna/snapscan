# Use a slim Python image
FROM python:3.9-slim

# Install system dependencies (same as before)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first (for better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 🔥 NEW: Set DeepFace home to a writable location inside the container
ENV DEEPFACE_HOME=/app/.deepface

# 🔥 NEW: Pre-download the Facenet model during build (saves ~2-3 minutes on first startup)
RUN python -c "from deepface import DeepFace; DeepFace.build_model('Facenet')"

# Copy the rest of the application
COPY . .

EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1", "app:app"]