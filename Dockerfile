FROM python:3.11-slim

WORKDIR /app

# Prevent Python from writing bytecode and ensure immediate log output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install curl for container healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and assets
COPY . /app/

# Make startup script executable
RUN chmod +x /app/start.sh

# Expose ports: 8000 (FastAPI Backend), 8501 (Streamlit Frontend)
EXPOSE 8000
EXPOSE 8501

# Entrypoint command
CMD ["/app/start.sh"]
