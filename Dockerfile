# Multi-stage build for Roof Detection app
# Stage 1: Build frontend
FROM node:20-alpine AS frontend-builder

WORKDIR /app/app
COPY app/package*.json ./
RUN npm ci
COPY app/ ./
RUN npm run build

# Stage 2: Python backend with frontend static files
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY api/ ./api/
COPY geocode_satellite.py ./
COPY inference.py ./

# Copy built frontend from stage 1
COPY --from=frontend-builder /app/app/dist ./app/dist

# Create models directory
RUN mkdir -p models

# Environment variables
ENV MODEL_PATH=/app/models/best.pt
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Run the server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
