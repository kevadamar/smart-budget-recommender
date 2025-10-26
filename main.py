from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import traceback
import os

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from models import (
    BudgetRequest, BudgetResponse, ErrorResponse, HealthResponse,
    ModelInfo, ModelType
)
from inference import predictor

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app
app = FastAPI(
    title="Budget Recommendation API",
    description="API for predicting household budget allocations using ML models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add rate limit middleware
app.add_middleware(SlowAPIMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limit exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    error_response = ErrorResponse(
        error=str(exc),
        details={"traceback": traceback.format_exc()} if os.getenv("DEBUG") else None
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.dict()
    )

@app.get("/", tags=["Root"])
@limiter.limit("100/minute")
async def root(request: Request):
    """Root endpoint"""
    return {
        "message": "Budget Recommendation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
@limiter.limit("100/minute")
async def health_check(request: Request):
    """Check API health and model status"""
    model_info = predictor.get_model_info()

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        models_loaded=model_info["models_loaded"],
        model_types=model_info.get("model_types", {}),
        version="1.0.0"
    )

@app.get("/models/info", response_model=ModelInfo, tags=["Models"])
@limiter.limit("100/minute")
async def get_model_info(request: Request):
    """Get information about loaded models"""
    info = predictor.get_model_info()

    return ModelInfo(
        model_type="ensemble",
        categories=info["categories"],
        feature_names=info["feature_names"],
        performance_metrics=None  # Could be added if saved during training
    )

@app.post("/predict", response_model=BudgetResponse, tags=["Prediction"])
@limiter.limit("100/minute")
async def predict_budget(request: Request, budget_request: BudgetRequest):
    """
    Predict budget allocation for given household parameters

    - **income**: Monthly household income in IDR
    - **family_size**: Number of family members
    - **vehicles**: Number of vehicles owned (default: 0)
    - **months_ahead**: Number of months to predict (1-3, default: 1)
    - **use_inflation**: Apply inflation adjustment (default: True)
    - **start_month**: Start month in YYYY-MM format (optional)
    - **model_type**: Type of model to use (xgboost, ensemble, best)
    """
    try:
        # Validate request
        if budget_request.income <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Income must be greater than 0"
            )

        if budget_request.family_size < 1 or budget_request.family_size > 20:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Family size must be between 1 and 20"
            )

        if budget_request.months_ahead < 1 or budget_request.months_ahead > 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Months ahead must be between 1 and 3"
            )

        # Generate predictions
        predictions = predictor.predict_budget(budget_request)

        # Calculate summary statistics
        total_expenses = [p.total_expense for p in predictions]
        total_savings = [p.savings for p in predictions]

        summary = {
            "average_monthly_expense": sum(total_expenses) / len(total_expenses),
            "average_monthly_savings": sum(total_savings) / len(total_savings),
            "total_months": len(predictions),
            "total_predicted_expense": sum(total_expenses),
            "total_predicted_savings": sum(total_savings),
            "average_allocation": {
                "primer_percentage": sum(p.breakdown.primer_percentage for p in predictions) / len(predictions),
                "sekunder_percentage": sum(p.breakdown.sekunder_percentage for p in predictions) / len(predictions),
                "tersier_percentage": sum(p.breakdown.tersier_percentage for p in predictions) / len(predictions)
            }
        }

        # Add metadata
        metadata = {
            "request_params": budget_request.dict(),
            "prediction_timestamp": datetime.now().isoformat(),
            "model_info": predictor.get_model_info()
        }

        return BudgetResponse(
            success=True,
            data=predictions,
            summary=summary,
            metadata=metadata
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/predict/sample", response_model=BudgetResponse, tags=["Prediction"])
@limiter.limit("100/minute")
async def predict_sample(request: Request):
    """Generate a sample prediction for testing"""
    sample_request = BudgetRequest(
        income=10000000,  # 10 million IDR
        family_size=4,
        vehicles=2,
        months_ahead=3,
        use_inflation=True,
        start_month="2025-01",
        model_type=ModelType.best
    )

    # Call predict_budget directly with the request and budget_request
    try:
        # Generate predictions
        predictions = predictor.predict_budget(sample_request)

        # Calculate summary statistics
        total_expenses = [p.total_expense for p in predictions]
        total_savings = [p.savings for p in predictions]

        summary = {
            "average_monthly_expense": sum(total_expenses) / len(total_expenses),
            "average_monthly_savings": sum(total_savings) / len(total_savings),
            "total_months": len(predictions),
            "total_predicted_expense": sum(total_expenses),
            "total_predicted_savings": sum(total_savings),
            "average_allocation": {
                "primer_percentage": sum(p.breakdown.primer_percentage for p in predictions) / len(predictions),
                "sekunder_percentage": sum(p.breakdown.sekunder_percentage for p in predictions) / len(predictions),
                "tersier_percentage": sum(p.breakdown.tersier_percentage for p in predictions) / len(predictions)
            }
        }

        # Add metadata
        metadata = {
            "request_params": sample_request.dict(),
            "prediction_timestamp": datetime.now().isoformat(),
            "model_info": predictor.get_model_info()
        }

        return BudgetResponse(
            success=True,
            data=predictions,
            summary=summary,
            metadata=metadata
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sample prediction failed: {str(e)}"
        )

@app.get("/categories", tags=["Reference"])
@limiter.limit("100/minute")
async def get_budget_categories(request: Request):
    """Get list of budget categories used by the model"""
    info = predictor.get_model_info()

    return {
        "primer_categories": info["primer_categories"],
        "sekunder_categories": info["sekunder_categories"],
        "tersier_categories": info["tersier_categories"],
        "all_categories": [cat.replace('_expenditure', '').replace('_', ' ').title()
                          for cat in info["categories"]]
    }

@app.get("/features", tags=["Reference"])
@limiter.limit("100/minute")
async def get_feature_info(request: Request):
    """Get information about features used by the model"""
    info = predictor.get_model_info()

    return {
        "features": info["feature_names"],
        "description": {
            "total_income": "Total monthly household income",
            "family_size": "Number of family members",
            "total_vehicles": "Number of vehicles owned",
            "income_per_capita": "Income per family member",
            "vehicles_per_capita": "Vehicles per family member",
            "region_encoded": "Encoded region information"
        }
    }

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )