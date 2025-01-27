import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Define paths
DATA_PATH = '/Users/damian/Documents/PJATK/Przedmioty/2025/SUML/SUML Projekt/dataset/processed_apartments_data.csv'
MODELS_DIR = '/Users/damian/Documents/PJATK/Przedmioty/2025/SUML/SUML Projekt/dataset/models'

class ModelTrainer:
    def __init__(self, data_path: str):
        """Initialize ModelTrainer with data path"""
        self.data_path = data_path
        self.models_dir = MODELS_DIR
        self.ensure_model_directory()
        
    def ensure_model_directory(self):
        """Create models directory if it doesn't exist"""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def load_data(self):
        """Load and prepare data for modeling"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        
        # Separate features and target
        self.X = self.df.drop(['id', 'price'], axis=1)
        self.y = self.df['price']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Scale numerical features
        self.scaler = StandardScaler()
        self.numerical_features = ['squareMeters', 'rooms', 'buildYear', 'centreDistance']
        
        self.X_train_scaled = self.X_train.copy()
        self.X_test_scaled = self.X_test.copy()
        
        self.X_train_scaled[self.numerical_features] = self.scaler.fit_transform(
            self.X_train[self.numerical_features]
        )
        self.X_test_scaled[self.numerical_features] = self.scaler.transform(
            self.X_test[self.numerical_features]
        )
        
        print("Data loaded and preprocessed successfully")
        
        # Save feature names for later use
        self.feature_names = self.X_train.columns.tolist()

    def evaluate_model(self, model, X_test, y_test, model_name: str):
        """Evaluate model performance"""
        predictions = model.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        metrics = {
            'Model': model_name,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'RMSE (PLN)': rmse,
            'MAE (PLN)': mae
        }
        
        return metrics, predictions

    def train_and_evaluate_models(self):
        """Train and evaluate multiple models"""
        print("\nTraining and evaluating models...")
        
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=15,
                min_samples_split=5,
                random_state=42
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=7,
                learning_rate=0.1,
                random_state=42
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=7,
                learning_rate=0.1,
                random_state=42
            ),
            'CatBoost': CatBoostRegressor(
                iterations=100,
                depth=7,
                learning_rate=0.1,
                random_seed=42,
                verbose=False
            )
        }
        
        results = []
        self.predictions = {}
        self.trained_models = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(self.X_train_scaled, self.y_train)
            
            # Save trained model
            self.trained_models[name] = model
            
            # Evaluate on test set
            metrics, preds = self.evaluate_model(
                model, self.X_test_scaled, self.y_test, name
            )
            results.append(metrics)
            self.predictions[name] = preds
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, self.X_train_scaled, self.y_train, 
                cv=5, scoring='r2'
            )
            print(f"Cross-validation R2 scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.results_df = pd.DataFrame(results)
        return self.results_df

    def analyze_feature_importance(self):
        """Analyze feature importance of the Random Forest model"""
        rf_model = self.trained_models['Random Forest']
        importances = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf_model.feature_importances_
        })
        importances = importances.sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        plt.bar(importances['feature'], importances['importance'])
        plt.xticks(rotation=45, ha='right')
        plt.title('Feature Importance (Random Forest)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.models_dir, 'feature_importance.png'))
        plt.close()
        
        print("\nTop 10 most important features:")
        print(importances.head(10))

    def analyze_price_predictions(self):
        """Analyze actual vs predicted prices"""
        rf_predictions = self.predictions['Random Forest']
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, rf_predictions, alpha=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 
                'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Actual vs Predicted Prices (Random Forest)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.models_dir, 'price_predictions.png'))
        plt.close()

    def compare_city_predictions(self):
        """Compare predictions for the same apartment in different cities"""
        # Create a sample apartment
        sample_apartment = pd.DataFrame({
            'squareMeters': [50],
            'rooms': [2],
            'buildYear': [2010],
            'centreDistance': [5],
            'hasParkingSpace': [True],
            'hasBalcony': [True],
            'hasElevator': [True],
            'hasSecurity': [True],
            'hasStorageRoom': [True]
        })
        
        # Add scaled features
        sample_scaled = sample_apartment.copy()
        sample_scaled[self.numerical_features] = self.scaler.transform(
            sample_apartment[self.numerical_features]
        )
        
        # Create predictions for each city
        cities = ['warszawa', 'krakow', 'wroclaw', 'gdansk', 'lodz']
        predictions = []
        
        for city in cities:
            city_apartment = sample_scaled.copy()
            # Add city columns (one-hot encoding)
            for c in cities:
                city_apartment[f'city_{c}'] = 1 if c == city else 0
            
            # Add age category columns
            age = 2025 - sample_apartment['buildYear'].values[0]
            city_apartment['age_new'] = 0
            city_apartment['age_contemporary'] = 0
            city_apartment['age_older'] = 1
            city_apartment['age_old'] = 0
            
            # Add zone columns
            city_apartment['zone_centrum'] = 0
            city_apartment['zone_bliska_strefa'] = 1
            city_apartment['zone_srednia_strefa'] = 0
            city_apartment['zone_peryferia'] = 0
            
            # Make prediction
            rf_model = self.trained_models['Random Forest']
            pred = rf_model.predict(city_apartment[self.feature_names])[0]
            predictions.append({'City': city.capitalize(), 'Predicted Price': pred})
        
        predictions_df = pd.DataFrame(predictions)
        print("\nPredicted prices for a sample apartment (50m², 2 rooms, built in 2010, 5km from center):")
        print(predictions_df)
        
        # Save predictions
        predictions_df.to_csv(os.path.join(self.models_dir, 'city_predictions.csv'), index=False)

    def save_best_model(self):
        """Save the best performing model"""
        best_model_name = self.results_df.loc[self.results_df['R2'].idxmax(), 'Model']
        print(f"\nBest model: {best_model_name}")
        
        # Save model comparison results
        comparison_path = os.path.join(self.models_dir, 'model_comparison.csv')
        self.results_df.to_csv(comparison_path, index=False)
        print(f"Model comparison saved to: {comparison_path}")
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'Actual': self.y_test,
            **{f'{name}_pred': pred for name, pred in self.predictions.items()}
        })
        predictions_path = os.path.join(self.models_dir, 'predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Predictions saved to: {predictions_path}")
        
        # Save best model and scaler
        best_model = self.trained_models[best_model_name]
        joblib.dump(best_model, os.path.join(self.models_dir, 'best_model.joblib'))
        joblib.dump(self.scaler, os.path.join(self.models_dir, 'scaler.joblib'))

def main():
    # Initialize trainer
    trainer = ModelTrainer(DATA_PATH)
    
    # Load and prepare data
    trainer.load_data()
    
    # Train and evaluate models
    results = trainer.train_and_evaluate_models()
    
    # Print results
    print("\nModel Comparison Results:")
    print(results.to_string())
    
    # Analyze Random Forest model
    trainer.analyze_feature_importance()
    trainer.analyze_price_predictions()
    trainer.compare_city_predictions()
    
    # Save best model and results
    trainer.save_best_model()

if __name__ == "__main__":
    main()