FROM python:3.10-slim

WORKDIR /app

# 1) System deps: certificates (for HTTPS to PyPI) + curl (optional, for debugging)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 2) Install Python deps
COPY requirements.inference.txt /app/requirements.inference.txt
RUN pip install --no-cache-dir -r requirements.inference.txt

# 3) Copy app code
COPY app /app/app
COPY src /app/src
COPY configs /app/configs
COPY models/best /app/models/best

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
