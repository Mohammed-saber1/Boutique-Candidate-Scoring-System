FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Train model (generates model.joblib)
RUN python -m app.models.train

# Default: score the sample candidate
ENTRYPOINT ["python", "-m", "app.main"]
CMD ["--input", "data/sample_candidate.json"]
