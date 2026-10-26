# Multi-purpose Docker image for Credit Portfolio Segmentation & Risk Pattern Analysis
# Supports both Streamlit Dashboard (Port 8501) and JupyterLab (Port 8888)

FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install minimal OS dependencies needed for compiling numerical wheels & healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Leverage Docker cache layer for Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code, dashboard, notebooks, data, and tests
COPY src/ /app/src/
COPY dashboard/ /app/dashboard/
COPY notebooks/ /app/notebooks/
COPY data/ /app/data/
COPY outputs/ /app/outputs/
COPY tests/ /app/tests/
COPY README.md /app/README.md

# Expose ports for Streamlit (8501) and JupyterLab (8888)
EXPOSE 8501 8888

# Default command: Launch the interactive Streamlit dashboard
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
