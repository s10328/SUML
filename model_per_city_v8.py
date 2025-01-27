import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional
import logging
from itertools import combinations

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CityPropertyModelTrainer:
    def __init__(self, data_path: str, city_name: str, output_dir: str):
        self.data_path = data_path
        self.city_name = city_name.lower()
        self.output_dir = os.path.join(output_dir, self.city_name)
        self.numeric_features = ['squareMeters', 'rooms', 'buildYear', 'centreDistance']
        self.amenity_features = ['hasParkingSpace', 'hasBalcony', 'hasElevator', 
                               'hasSecurity', 'hasStorageRoom']
        
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception as e:
            logging.error(f"Nie można utworzyć katalogu dla miasta {self.city_name}: {e}")
            raise
    


    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        try:
            logging.info(f"Wczytywanie danych dla miasta {self.city_name.capitalize()}...")
            df = pd.read_csv(self.data_path)
            
            city_mask = df[f'city_{self.city_name}'] == True
            df = df[city_mask]
            
            
            self.X = df.drop(['id', 'price'] + 
                           [col for col in df.columns if col.startswith('city_')], 
                           axis=1)
            self.y = df['price']
            
            return self.X, self.y
            
        except Exception as e:
            logging.error(f"Błąd podczas wczytywania danych dla {self.city_name}: {e}")
            raise

    def create_pipeline(self) -> Tuple[Pipeline, Dict]:
        param_grid = {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__max_depth': [10, 15, 20, None],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4],
            'regressor__max_features': ['sqrt', 'log2'],
            'regressor__bootstrap': [True, False]
        }
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
        ])
        
        return pipeline, param_grid

    def analyze_price_distribution(self):
        """
        Analizuje i wizualizuje rozkład cen w mieście.
        """
        plt.figure(figsize=(10, 6))
        plt.hist(self.y, bins=50, edgecolor='black')
        plt.title(f'Rozkład cen nieruchomości w mieście {self.city_name.capitalize()}')
        plt.xlabel('Cena (PLN)')
        plt.ylabel('Liczba nieruchomości')
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f'price_distribution.png')
        plt.savefig(output_path)
        plt.close()

    def train_and_evaluate(self) -> GridSearchCV:
        """
        Trenuje i ocenia model dla miasta.
        """
        try:
            # Podział danych
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )
            
            pipeline, param_grid = self.create_pipeline()
            
            # Grid Search
            grid_search = GridSearchCV(
                pipeline, 
                param_grid, 
                cv=5, 
                scoring='r2', 
                n_jobs=-1,
                verbose=1
            )
            
            logging.info(f"\nRozpoczęcie treningu modelu dla {self.city_name.capitalize()}...")
            grid_search.fit(self.X_train, self.y_train)
            logging.info("Trening zakończony")
            
            self.model = grid_search.best_estimator_
            
            # Najlepsze parametry
            logging.info("\nNajlepsze parametry:")
            logging.info(grid_search.best_params_)
            
            # Predykcja
            y_pred = grid_search.predict(self.X_test)
            
            # Analiza modelu
            self.evaluate_model(self.y_test, y_pred)
            self.analyze_feature_importance(grid_search)
            self.analyze_predictions(self.y_test, y_pred)
            self.analyze_price_ranges(self.y_test, y_pred)
            self.analyze_price_distribution()
            
            return grid_search
            
        except Exception as e:
            logging.error(f"Błąd podczas treningu modelu dla {self.city_name}: {e}")
            raise

    def evaluate_model(self, y_test: pd.Series, y_pred: np.ndarray) -> Dict:
        """
        Ocenia wydajność modelu używając różnych metryk.
        """
        metrics = {
            'R2 Score': r2_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAPE': mean_absolute_percentage_error(y_test, y_pred) * 100
        }
        
        # Cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, self.X, self.y, cv=cv, scoring='r2')
        
        metrics['CV Mean R2'] = cv_scores.mean()
        metrics['CV Std R2'] = cv_scores.std()
        
        logging.info(f"\nMetryki wydajności modelu dla {self.city_name.capitalize()}:")
        for metric, value in metrics.items():
            logging.info(f"{metric}: {value:.4f}")
        
        return metrics

    def analyze_feature_importance(self, grid_search: GridSearchCV) -> pd.DataFrame:
        """
        Analizuje i wizualizuje ważność cech.
        """
        try:
            feature_names = self.X.columns
            importances = grid_search.best_estimator_.named_steps['regressor'].feature_importances_
            
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

            logging.info(f"\nNajważniejsze cechy dla {self.city_name.capitalize()}:")
            logging.info(feature_importance)

            plt.figure(figsize=(12, 6))
            plt.bar(feature_importance['feature'], feature_importance['importance'])
            plt.xticks(rotation=45, ha='right')
            plt.title(f'Ważność cech dla {self.city_name.capitalize()}')
            plt.tight_layout()
            
            output_path = os.path.join(self.output_dir, f'feature_importance.png')
            plt.savefig(output_path)
            plt.close()

            return feature_importance
            
        except Exception as e:
            logging.error(f"Błąd podczas analizy ważności cech dla {self.city_name}: {e}")
            raise

    def analyze_predictions(self, y_test: pd.Series, y_pred: np.ndarray) -> Dict:
        """
        Analizuje jakość predykcji modelu.
        """
        try:
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot(
                [y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', 
                lw=2
            )
            plt.xlabel('Rzeczywista cena')
            plt.ylabel('Przewidywana cena')
            plt.title(f'Rzeczywiste vs Przewidywane ceny dla {self.city_name.capitalize()}')
            plt.tight_layout()
            
            output_path = os.path.join(self.output_dir, f'predictions_scatter.png')
            plt.savefig(output_path)
            plt.close()
            
            errors = y_pred - y_test
            error_stats = {
                'Średni błąd': errors.mean(),
                'Odchylenie standardowe błędu': errors.std(),
                'Maksymalne przeszacowanie': errors.max(),
                'Maksymalne niedoszacowanie': errors.min()
            }
            
            logging.info(f"\nAnaliza błędów predykcji dla {self.city_name.capitalize()}:")
            for stat, value in error_stats.items():
                logging.info(f"{stat}: {value:,.2f} PLN")
                
            return error_stats
            
        except Exception as e:
            logging.error(f"Błąd podczas analizy predykcji dla {self.city_name}: {e}")
            raise

    def analyze_price_ranges(self, y_test: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
        """
        Analizuje wydajność modelu w różnych zakresach cenowych.
        """
        try:
            results_df = pd.DataFrame({
                'Actual': y_test,
                'Predicted': y_pred,
                'Error': y_pred - y_test,
                'Percentage_Error': ((y_pred - y_test) / y_test) * 100
            })
            
            # Podział na kwartyle
            price_ranges = pd.qcut(results_df['Actual'], q=4)
            
            logging.info(f"\nAnaliza wydajności w przedziałach cenowych dla {self.city_name.capitalize()}:")
            for price_range in price_ranges.unique():
                range_df = results_df[price_ranges == price_range]
                logging.info(f"\nPrzedział {price_range}:")
                logging.info(f"Liczba nieruchomości: {len(range_df)}")
                logging.info(f"MAPE: {abs(range_df['Percentage_Error']).mean():.2f}%")
                logging.info(f"Średni błąd: {range_df['Error'].mean():,.2f} PLN")
                logging.info(f"R2 Score: {r2_score(range_df['Actual'], range_df['Predicted']):.4f}")
            
            return results_df
            
        except Exception as e:
            logging.error(f"Błąd podczas analizy zakresów cenowych dla {self.city_name}: {e}")
            raise

    def save_model(self, grid_search: GridSearchCV) -> None:
        """
        Zapisuje wytrenowany model do pliku.
        """
        try:
            model_path = os.path.join(self.output_dir, f'model.joblib')
            joblib.dump(grid_search.best_estimator_, model_path)
            logging.info(f"\nModel dla {self.city_name.capitalize()} został zapisany w: {model_path}")
            
        except Exception as e:
            logging.error(f"Błąd podczas zapisywania modelu dla {self.city_name}: {e}")
            raise

def main():
    try:
        data_path = '/Users/damian/Documents/PJATK/Przedmioty/2025/SUML/SUML Projekt/dataset/processed_apartments_data.csv'
        output_dir = '/Users/damian/Documents/PJATK/Przedmioty/2025/SUML/SUML Projekt/models_by_city_v8_final'
        
        cities = ['warszawa', 'krakow', 'lodz', 'wroclaw', 'gdansk']
        city_models = {}
        
        for city in cities:
            logging.info(f"\n{'='*50}")
            logging.info(f"Rozpoczęcie treningu modelu dla miasta {city.capitalize()}")
            logging.info(f"{'='*50}")
            
            trainer = CityPropertyModelTrainer(
                data_path=data_path,
                city_name=city,
                output_dir=output_dir
            )
            
            trainer.load_and_prepare_data()
            grid_search = trainer.train_and_evaluate()
            trainer.save_model(grid_search)
            
            city_models[city] = grid_search
            
        logging.info("\nTrening wszystkich modeli zakończony pomyślnie!")
        
    except Exception as e:
        logging.error(f"Wystąpił błąd podczas wykonywania głównej funkcji: {e}")
        raise

if __name__ == "__main__":
    main()