from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class ModelType(str, Enum):
    xgboost = "xgboost"
    ensemble = "ensemble"
    best = "best"

class BudgetRequest(BaseModel):
    """Request model for budget prediction"""
    income: float = Field(..., gt=0, description="Monthly household income in IDR")
    family_size: int = Field(..., ge=1, le=20, description="Number of family members")
    vehicles: int = Field(default=0, ge=0, le=10, description="Number of vehicles owned")
    months_ahead: int = Field(default=1, ge=1, le=3, description="Number of months to predict ahead")
    use_inflation: bool = Field(default=True, description="Whether to apply inflation adjustment")
    start_month: Optional[str] = Field(None, description="Start month in YYYY-MM format")
    model_type: ModelType = Field(ModelType.best, description="Type of model to use for prediction")

class BudgetCategory(BaseModel):
    """Single budget category prediction"""
    name: str
    amount: float
    percentage: float

class BudgetBreakdown(BaseModel):
    """Detailed budget breakdown"""
    primer_categories: List[BudgetCategory]
    sekunder_categories: List[BudgetCategory]
    tersier_categories: List[BudgetCategory]
    total_primer: float
    total_sekunder: float
    total_tersier: float
    primer_percentage: float
    sekunder_percentage: float
    tersier_percentage: float

class MonthlyPrediction(BaseModel):
    """Single month budget prediction"""
    month: str
    month_name: str
    income: float
    total_expense: float
    savings: float
    savings_percentage: float
    breakdown: BudgetBreakdown
    cumulative_inflation_pct: float = 0.0
    inflation_adjusted: bool = False

class BudgetResponse(BaseModel):
    """Response model for budget prediction"""
    success: bool
    data: List[MonthlyPrediction]
    summary: Dict[str, Any]
    metadata: Dict[str, Any]

class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = False
    error: str
    details: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    models_loaded: bool
    model_types: Dict[str, str]
    version: str = "1.0.0"

class ModelInfo(BaseModel):
    """Model information response"""
    model_type: str
    categories: List[str]
    feature_names: List[str]
    last_updated: Optional[datetime] = None
    performance_metrics: Optional[Dict[str, float]] = None