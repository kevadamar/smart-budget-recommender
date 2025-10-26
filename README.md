# Smart Budget Recommender

A machine learning-based budget recommendation system that predicts optimal household budget allocations across different expense categories based on family income and demographics.

## 📊 Project Overview

This project analyzes family income and expenditure patterns to provide intelligent budget recommendations. It uses machine learning models (XGBoost and Neural Networks) trained on Indonesian family expenditure data to predict optimal budget allocations across primary, secondary, and tertiary expense categories.

### Key Features
- **Data-driven Recommendations**: Uses ML models trained on real Indonesian family expenditure data
- **Multi-category Budgeting**: Categorizes expenses into primer, sekunder, and tersier categories
- **Inflation Adjustment**: Supports multi-month predictions with inflation considerations
- **RESTful API**: Easy integration with web and mobile applications
- **Ensemble Models**: Combines XGBoost and Neural Network models for improved accuracy

## 📁 Project Structure

```
smart-budget-ml/
├── api/                     # FastAPI inference API
│   ├── main.py             # API endpoints and application logic
│   ├── model_loader.py     # Model loading and prediction logic
│   ├── requirements.txt    # API dependencies
│   └── Dockerfile          # Docker configuration
├── models/                 # Trained ML models
├── data/                   # Dataset and processed data
├── frontend/               # Frontend application (if available)
├── Capstone Dicoding EDA.ipynb  # Exploratory Data Analysis
├── inference.py           # Standalone inference script
└── README.md              # This file
```

## 🔍 Data Analysis & Model Development

### Exploratory Data Analysis (EDA)
The [Capstone Dicoding EDA.ipynb](Capstone%20Dicoding%20EDA.ipynb) notebook contains comprehensive data analysis including:

1. **Data Understanding**: Structure analysis of the Family Income and Expenditure dataset
2. **Statistical Analysis**: Distribution analysis of income and expenditure patterns
3. **Category Classification**: Categorizing expenses into:
   - **Primer Categories**: Food, Housing, Transportation, Communication, Medical, Education
   - **Sekunder Categories**: Clothing, Restaurant & Hotels, Miscellaneous Goods
   - **Tersier Categories**: Alcoholic Beverages, Tobacco, Special Occasions
4. **Correlation Analysis**: Understanding relationships between expenditure categories
5. **Inflation Data Integration**: Web scraping BPS Indonesia for monthly inflation rates

### Machine Learning Pipeline
- **Feature Engineering**: Creating relevant features from raw income and demographic data
- **Model Training**: Training separate models for each expenditure category
- **Ensemble Methods**: Combining XGBoost and Neural Network models
- **Hyperparameter Tuning**: Optimizing model performance using GridSearchCV
- **Model Evaluation**: Comparing model performance using MAPE, RMSE, and R² metrics

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Trained models in the `models/` directory

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd smart-budget-ml
```

2. **Install API dependencies**
```bash
cd api
pip install -r requirements.txt
```

3. **Prepare Models**
Ensure your trained models are in the `../models/` directory:
- `best_models_final.pkl` - Best performing models
- `best_model_types.pkl` - Model type information
- `best_xgboost_models.pkl` - XGBoost models
- `best_ensemble_models.pkl` - Ensemble models
- `feature_scaler.pkl` - Feature scaler

### Running the API

```bash
# Development mode
cd api
uvicorn main:app --reload

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Access Points
- **API Documentation**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 📡 API Documentation

### Endpoints

#### POST /predict
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

#### GET /predict/sample
Generate a sample prediction for testing.

#### GET /health
Check API health and model status.

#### GET /models/info
Get information about loaded models.

#### GET /categories
Get list of budget categories used by the model.

#### GET /features
Get information about features used by the model.

### Model Types
- **xgboost**: Use XGBoost models only
- **ensemble**: Use ensemble models (XGBoost + Neural Network)
- **best**: Use the best performing model for each category (recommended)

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t budget-api ./api
```

### Run Docker Container
```bash
docker run -d -p 8000:8000 -v $(pwd)/models:/app/models budget-api
```

### Docker Compose
```yaml
version: '3.8'
services:
  budget-api:
    build: ./api
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - PYTHONPATH=/app
      - DEBUG=false
```

```bash
docker-compose up -d
```

## 🧪 Testing

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

## 💡 Usage Examples

### Standalone Inference
```bash
python inference.py --income 10000000 --family_size 4 --vehicles 2
```

### Batch Predictions
```python
from api.model_loader import BudgetPredictor

predictor = BudgetPredictor()
results = predictor.predict_multiple_months(
    income=15000000,
    family_size=5,
    vehicles=1,
    months_ahead=6
)
```

## 📈 Budget Categories

### Primer Categories (Essential Needs)
- Total Food Expenditure
- Housing and Water Expenditure
- Transportation Expenditure
- Communication Expenditure
- Medical Care Expenditure
- Education Expenditure

### Sekunder Categories (Lifestyle)
- Clothing, Footwear and Other Wear Expenditure
- Restaurant and Hotels Expenditure
- Miscellaneous Goods and Services Expenditure

### Tersier Categories (Non-essential)
- Alcoholic Beverages Expenditure
- Tobacco Expenditure
- Special Occasions Expenditure

## 🔧 Development

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

## 📊 Model Performance

The ensemble models achieve the following performance metrics:
- **MAPE**: < 15% on average across categories
- **RMSE**: Varies by category based on expenditure scale
- **R²**: > 0.85 for most primer categories

## 🌐 Environment Variables

- `DEBUG`: Enable debug mode (default: false)
- `PYTHONPATH`: Python path (default: /app)
- `PYTHONUNBUFFERED`: Disable output buffering (default: 1)

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support

For support and questions, please open an issue in the repository.