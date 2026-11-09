FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY docuchat ./docuchat
RUN pip install --no-cache-dir .

COPY sample_docs ./sample_docs
# Pre-load the sample document so the app has something to search on first start.
RUN python -m docuchat ingest sample_docs/brightwave_handbook.pdf

EXPOSE 8000
CMD ["uvicorn", "docuchat.api:app", "--host", "0.0.0.0", "--port", "8000"]
