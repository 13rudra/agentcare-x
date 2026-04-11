FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models.py .
COPY openenv.yaml .
COPY inference.py .
COPY __init__.py .
COPY pyproject.toml .
COPY uv.lock .
COPY server/ ./server/
COPY env/ ./env/
COPY tasks/ ./tasks/
COPY graders/ ./graders/
COPY dashboard/ ./dashboard/

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
