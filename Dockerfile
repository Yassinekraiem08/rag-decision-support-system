FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY docs/ ./docs/
COPY init_db.py .
COPY ingest_folder.py .
COPY streamlit_app.py .

# Never copy .env — pass secrets via environment variables at runtime

# Expose port
EXPOSE 8000

# Run FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
