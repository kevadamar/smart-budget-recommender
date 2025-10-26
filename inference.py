import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings("ignore")

from models import BudgetRequest, BudgetBreakdown, MonthlyPrediction, ModelType

class BudgetPredictor:
    """Budget prediction service using trained ML models"""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.scaler = None
        self.feature_names = []
        self.best_models = {}
        self.best_model_types = {}
        self.xgb_models = {}
        self.ensemble_models = {}

        # Define budget categories
        self.primer = [
            "total_food_expenditure",
            "housing_and_water_expenditure",
            "transportation_expenditure",
            "communication_expenditure",
            "medical_care_expenditure",
            "education_expenditure",
        ]

        self.sekunder = [
            "clothing_footwear_and_other_wear_expenditure",
            "restaurant_and_hotels_expenditure",
            "miscellaneous_goods_and_services_expenditure",
        ]

        self.tersier = [
            "alcoholic_beverages_expenditure",
            "tobacco_expenditure",
            "special_occasions_expenditure",
        ]

        self.all_categories = self.primer + self.sekunder + self.tersier

        # Load models on initialization
        self.load_models()

    def load_models(self):
        """Load all trained models and scalers"""
        try:
            # Load feature scaler
            scaler_path = os.path.join(self.models_dir, "feature_scaler.pkl")
            if os.path.exists(scaler_path):
                scaler_data = joblib.load(scaler_path)
                self.scaler = scaler_data['scaler']
                self.feature_names = scaler_data.get('feature_names', [])
                print(f"Loaded feature scaler from {scaler_path}")
            else:
                print(f"Feature scaler not found at {scaler_path}")

            # Load best models
            best_models_path = os.path.join(self.models_dir, "best_models_final.pkl")
            if os.path.exists(best_models_path):
                self.best_models = joblib.load(best_models_path)
                print(f"Loaded best models from {best_models_path}")

            # Load model types
            model_types_path = os.path.join(self.models_dir, "best_model_types.pkl")
            if os.path.exists(model_types_path):
                self.best_model_types = joblib.load(model_types_path)
                print(f"Loaded model types from {model_types_path}")

            # Load XGBoost models
            xgb_path = os.path.join(self.models_dir, "best_xgboost_models.pkl")
            if os.path.exists(xgb_path):
                self.xgb_models = joblib.load(xgb_path)
                print(f"Loaded XGBoost models from {xgb_path}")

            # Load ensemble models
            ensemble_path = os.path.join(self.models_dir, "best_ensemble_models.pkl")
            if os.path.exists(ensemble_path):
                self.ensemble_models = joblib.load(ensemble_path)
                print(f"Loaded ensemble models from {ensemble_path}")

            if not self.best_models:
                # Fallback: try to load individual model files
                self._load_individual_models()

            print(f"Models loaded successfully. Categories: {len(self.all_categories)}")

        except Exception as e:
            print(f"Error loading models: {str(e)}")
            self._create_dummy_models()

    def _load_individual_models(self):
        """Try to load individual model files"""
        model_files = [
            "budget_model_best.keras",
            "enhanced_budget_transfer_model.keras",
            "transfer_budget_model.keras",
            "transfer_model_final.keras",
            "transfer_model_optimal.keras"
        ]

        for model_file in model_files:
            model_path = os.path.join(self.models_dir, model_file)
            if os.path.exists(model_path):
                try:
                    # Try loading as Keras model
                    import tensorflow as tf
                    model = tf.keras.models.load_model(model_path)
                    # Use for all categories as fallback
                    for category in self.all_categories:
                        if category not in self.best_models:
                            self.best_models[category] = model
                            self.best_model_types[category] = "keras"
                    print(f"Loaded Keras model from {model_file}")
                    break
                except:
                    print(f"✗ Failed to load {model_file}")

    def _create_dummy_models(self):
        """Create dummy models for testing when no trained models are available"""
        print("Creating dummy models for testing...")
        from sklearn.dummy import DummyRegressor

        for category in self.all_categories:
            dummy = DummyRegressor(strategy="mean")
            # Fit with dummy data
            dummy.fit(np.array([[10000000, 4, 0, 2500000, 0, 0]]), np.array([1000000]))
            self.best_models[category] = dummy
            self.best_model_types[category] = "dummy"

    def _prepare_features(self, income: float, family_size: int, vehicles: int) -> np.ndarray:
        """Prepare input features for prediction"""
        # Create feature DataFrame
        features = pd.DataFrame({
            'total_income': [income],
            'family_size': [family_size],
            'total_vehicles': [vehicles],
            'income_per_capita': [income / family_size],
            'vehicles_per_capita': [vehicles / family_size],
            'region_encoded': [0]  # Default value
        })

        # Ensure correct feature order
        feature_order = [
            'total_income',
            'family_size',
            'total_vehicles',
            'income_per_capita',
            'vehicles_per_capita',
            'region_encoded'
        ]

        features = features[feature_order]

        # Apply scaling if available
        if self.scaler:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features.values

        return features_scaled, features.values

    def _predict_category(self, category: str, features_scaled: np.ndarray,
                         features_raw: np.ndarray, model_type: ModelType) -> float:
        """Predict expenditure for a single category"""
        if category not in self.best_models:
            return 0.0

        model = self.best_models[category]

        try:
            # Determine which model to use
            if model_type == ModelType.best:
                # Use the best model type determined during training
                actual_model_type = self.best_model_types.get(category, "xgboost")
            elif model_type == ModelType.xgboost:
                actual_model_type = "xgboost"
            elif model_type == ModelType.ensemble:
                actual_model_type = "ensemble"
            else:
                actual_model_type = "xgboost"

            # Make prediction based on model type
            if actual_model_type == "xgboost":
                # XGBoost model prediction
                pred = model.predict(features_scaled)[0]
            elif actual_model_type == "ensemble":
                # Ensemble model - needs XGBoost prediction as additional feature
                if category in self.xgb_models:
                    xgb_pred = self.xgb_models[category].predict(features_scaled)[0]
                    # Combine original features with XGBoost prediction
                    ensemble_features = np.column_stack([features_scaled, [xgb_pred]])
                    pred = model.predict(ensemble_features)[0][0]
                else:
                    pred = model.predict(features_scaled)[0]
            elif actual_model_type == "keras":
                # Keras/TensorFlow model
                pred = model.predict(features_scaled)[0]
                if isinstance(pred, np.ndarray):
                    pred = float(pred[0] if len(pred.shape) > 0 else pred)
            else:
                # Dummy or sklearn model
                pred = model.predict(features_scaled)[0]

            # Ensure non-negative prediction
            return max(0, float(pred))

        except Exception as e:
            print(f"Error predicting {category}: {str(e)}")
            return 0.0

    def predict_budget(self, request: BudgetRequest) -> List[MonthlyPrediction]:
        """Generate budget predictions for given parameters"""
        # Prepare features
        features_scaled, features_raw = self._prepare_features(
            request.income, request.family_size, request.vehicles
        )

        # Set start month
        if request.start_month:
            start_date = datetime.strptime(request.start_month + '-01', '%Y-%m-%d')
        else:
            start_date = datetime.now()

        # Determine inflation rate
        inflation_rate = 0.21  # Default monthly inflation rate
        if request.use_inflation and os.path.exists("../bps_inflation_data.csv"):
            try:
                inflation_data = pd.read_csv("../bps_inflation_data.csv")
                if 'inflation_mom' in inflation_data.columns:
                    inflation_rate = inflation_data['inflation_mom'].mean()
            except:
                pass

        # Generate monthly predictions
        predictions = []

        for month_offset in range(request.months_ahead):
            # Calculate month date
            current_date = start_date + timedelta(days=month_offset * 30)
            month_label = current_date.strftime('%Y-%m')
            month_name = current_date.strftime('%B %Y')

            # Calculate cumulative inflation
            if month_offset > 0:
                cumulative_inflation = ((1 + inflation_rate/100) ** month_offset - 1) * 100
                inflation_factor = (1 + inflation_rate/100) ** month_offset
            else:
                cumulative_inflation = 0
                inflation_factor = 1.0

            # Predict base amounts for each category
            category_predictions = {}
            for category in self.all_categories:
                pred = self._predict_category(
                    category, features_scaled, features_raw, request.model_type
                )
                # Apply inflation if needed
                if request.use_inflation and month_offset > 0:
                    pred *= inflation_factor
                category_predictions[category] = pred

            # Calculate totals
            total_primer = sum(category_predictions[cat] for cat in self.primer)
            total_sekunder = sum(category_predictions[cat] for cat in self.sekunder)
            total_tersier = sum(category_predictions[cat] for cat in self.tersier)
            total_expense = total_primer + total_sekunder + total_tersier

            # Calculate percentages
            if total_expense > 0:
                primer_pct = (total_primer / total_expense) * 100
                sekunder_pct = (total_sekunder / total_expense) * 100
                tersier_pct = (total_tersier / total_expense) * 100
            else:
                primer_pct = sekunder_pct = tersier_pct = 0

            # Create budget breakdown
            primer_categories = []
            for cat in self.primer:
                amount = category_predictions[cat]
                pct = (amount / total_expense * 100) if total_expense > 0 else 0
                primer_categories.append({
                    "name": cat.replace('_expenditure', '').replace('_', ' ').title(),
                    "amount": amount,
                    "percentage": pct
                })

            sekunder_categories = []
            for cat in self.sekunder:
                amount = category_predictions[cat]
                pct = (amount / total_expense * 100) if total_expense > 0 else 0
                sekunder_categories.append({
                    "name": cat.replace('_expenditure', '').replace('_', ' ').title(),
                    "amount": amount,
                    "percentage": pct
                })

            tersier_categories = []
            for cat in self.tersier:
                amount = category_predictions[cat]
                pct = (amount / total_expense * 100) if total_expense > 0 else 0
                tersier_categories.append({
                    "name": cat.replace('_expenditure', '').replace('_', ' ').title(),
                    "amount": amount,
                    "percentage": pct
                })

            breakdown = BudgetBreakdown(
                primer_categories=primer_categories,
                sekunder_categories=sekunder_categories,
                tersier_categories=tersier_categories,
                total_primer=total_primer,
                total_sekunder=total_sekunder,
                total_tersier=total_tersier,
                primer_percentage=primer_pct,
                sekunder_percentage=sekunder_pct,
                tersier_percentage=tersier_pct
            )

            # Calculate savings
            savings = request.income - total_expense
            savings_pct = (savings / request.income * 100) if request.income > 0 else 0

            # Create monthly prediction
            prediction = MonthlyPrediction(
                month=month_label,
                month_name=month_name,
                income=request.income,
                total_expense=total_expense,
                savings=savings,
                savings_percentage=savings_pct,
                breakdown=breakdown,
                cumulative_inflation_pct=cumulative_inflation,
                inflation_adjusted=(month_offset > 0 and request.use_inflation)
            )

            predictions.append(prediction)

        return predictions

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            "categories": self.all_categories,
            "feature_names": self.feature_names or ["total_income", "family_size", "total_vehicles",
                                                   "income_per_capita", "vehicles_per_capita", "region_encoded"],
            "model_types": self.best_model_types,
            "models_loaded": len(self.best_models) > 0,
            "primer_categories": [cat.replace('_expenditure', '').replace('_', ' ').title()
                                 for cat in self.primer],
            "sekunder_categories": [cat.replace('_expenditure', '').replace('_', ' ').title()
                                   for cat in self.sekunder],
            "tersier_categories": [cat.replace('_expenditure', '').replace('_', ' ').title()
                                  for cat in self.tersier]
        }
        return info

# Global predictor instance
predictor = BudgetPredictor()