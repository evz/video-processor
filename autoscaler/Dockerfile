FROM python:3.11-slim

# Install Docker CLI
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    curl -fsSL https://get.docker.com | sh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir redis

COPY autoscale.py /app/autoscale.py

CMD ["python", "/app/autoscale.py"]
