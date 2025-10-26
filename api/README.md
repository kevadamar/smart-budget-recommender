# Budget Recommendation API

FastAPI-based inference API for household budget recommendation using trained ML models.

## Features

- Predict budget allocations across primer, sekunder, and tersier categories
- Support for multi-month predictions with inflation adjustment
- Multiple model types: XGBoost, Ensemble (XGBoost + Neural Network)
- RESTful API with automatic documentation
- Docker support for easy deployment

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Models

Ensure your trained models are in the `../models/` directory:
- `best_models_final.pkl` - Best performing models
- `best_model_types.pkl` - Model type information
- `best_xgboost_models.pkl` - XGBoost models
- `best_ensemble_models.pkl` - Ensemble models
- `feature_scaler.pkl` - Feature scaler

### 3. Run the API

```bash
# Development mode
uvicorn api.main:app --reload

# Production mode
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 4. Access the API

- API Documentation: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health Check: http://localhost:8000/health

## API Endpoints

### POST /predict

Generate budget predictions for given household parameters.

**Request Body:**
```json
{
  "income": 10000000,
  "family_size": 4,
  "vehicles": 2,
  "months_ahead": 3,
  "use_inflation": true,
  "start_month": "2025-01",
  "model_type": "best"
}
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "month": "2025-01",
      "month_name": "January 2025",
      "income": 10000000,
      "total_expense": 8500000,
      "savings": 1500000,
      "savings_percentage": 15.0,
      "breakdown": {
        "primer_categories": [...],
        "sekunder_categories": [...],
        "tersier_categories": [...]
      }
    }
  ],
  "summary": {...},
  "metadata": {...}
}
```

### GET /predict/sample

Generate a sample prediction for testing.

### GET /health

Check API health and model status.

### GET /models/info

Get information about loaded models.

### GET /categories

Get list of budget categories used by the model.

### GET /features

Get information about features used by the model.

## Model Types

- **xgboost**: Use XGBoost models only
- **ensemble**: Use ensemble models (XGBoost + Neural Network)
- **best**: Use the best performing model for each category (recommended)

## Budget Categories

### Primer Categories
- Total Food Expenditure
- Housing and Water Expenditure
- Transportation Expenditure
- Communication Expenditure
- Medical Care Expenditure
- Education Expenditure

### Sekunder Categories
- Clothing, Footwear and Other Wear Expenditure
- Restaurant and Hotels Expenditure
- Miscellaneous Goods and Services Expenditure

### Tersier Categories
- Alcoholic Beverages Expenditure
- Tobacco Expenditure
- Special Occasions Expenditure

## Docker Deployment

### Build Docker Image

```bash
docker build -t budget-api .
```

### Run Docker Container

```bash
docker run -d -p 8000:8000 -v $(pwd)/../models:/app/models budget-api
```

### Docker Compose

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  budget-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ../models:/app/models
    environment:
      - PYTHONPATH=/app
      - DEBUG=false
```

Run with Docker Compose:

```bash
docker-compose up -d
```

## Testing

### Test with curl

```bash
# Health check
curl http://localhost:8000/health

# Sample prediction
curl http://localhost:8000/predict/sample

# Custom prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "income": 15000000,
       "family_size": 5,
       "vehicles": 1,
       "months_ahead": 2
     }'
```

### Test with Python

```python
import httpx

# Sample prediction
response = httpx.get("http://localhost:8000/predict/sample")
print(response.json())

# Custom prediction
data = {
    "income": 12000000,
    "family_size": 4,
    "vehicles": 2,
    "months_ahead": 3,
    "use_inflation": True
}

response = httpx.post("http://localhost:8000/predict", json=data)
print(response.json())
```

## Development

### Code Formatting

```bash
black .
```

### Linting

```bash
flake8 .
```

### Running Tests

```bash
pytest
```

## Environment Variables

- `DEBUG`: Enable debug mode (default: false)
- `PYTHONPATH`: Python path (default: /app)
- `PYTHONUNBUFFERED`: Disable output buffering (default: 1)