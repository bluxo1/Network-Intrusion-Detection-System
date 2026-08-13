# =============================================================================
# NIDS-PyTorch container image
# Builds a CPU-only image that serves the Flask app with gunicorn.
# The trained model artifacts (models/*.pt, *.pkl) are copied in if present;
# otherwise train them first (see README) or mount them at runtime.
# =============================================================================
FROM python:3.11-slim

# Avoid interactive prompts and .pyc clutter; unbuffered logs for containers.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

WORKDIR /app

# Install CPU-only PyTorch first (smaller image), then the rest of the deps.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Copy the application source (see .dockerignore for exclusions).
COPY . .

EXPOSE 5000

# Simple container healthcheck against the Flask /health route.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:5000/health').status==200 else 1)"

# Production WSGI server. 4 workers is a sane default for small deployments.
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "--timeout", "120", "app.app:app"]
