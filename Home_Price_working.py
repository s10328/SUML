import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import os
import warnings

# Ignorowanie ostrzeżeń sklearn
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Konfiguracja strony
st.set_page_config(
    page_title="Home Price",
    page_icon="🏠",
    layout="wide"
)

# Ścieżka do modeli
MODELS_DIR = '/Users/damian/Documents/PJATK/Przedmioty/2025/SUML/SUML Projekt/models_by_city_v8_final'

def get_zone_from_distance(distance: float) -> str:
    """
    Określa strefę na podstawie odległości od centrum
    
    Args:
        distance: odległość od centrum w km
        
    Returns:
        str: nazwa strefy
    """
    if distance <= 2:
        return 'Centrum'
    elif distance <= 5:
        return 'Bliska strefa'
    elif distance <= 8:
        return 'Średnia strefa'
    else:
        return 'Peryferia'
    
def get_building_age_category(build_year: int, current_year: int = 2025) -> str:
    """
    Określa kategorię wieku budynku na podstawie roku budowy
    
    Args:
        build_year: rok budowy
        current_year: aktualny rok (domyślnie 2025)
        
    Returns:
        str: kategoria wieku budynku
    """
    age = current_year - build_year
    
    if age <= 5:
        return 'Nowy'
    elif age <= 20:
        return 'Współczesny'
    elif age <= 50:
        return 'Starszy'
    else:
        return 'Stary'

class RealEstatePredictor:
    def __init__(self):
        self.models = {}
        self.load_models()
        
    def load_models(self):
        """Ładuje modele dla wszystkich miast"""
        cities = {
            'warszawa': 'warszawa',
            'kraków': 'krakow',
            'łódź': 'lodz',
            'wrocław': 'wroclaw',
            'gdańsk': 'gdansk'
        }
        
        for city_pl, city_en in cities.items():
            model_path = os.path.join(MODELS_DIR, city_en, 'model.joblib')
            if os.path.exists(model_path):
                try:
                    self.models[city_pl] = joblib.load(model_path)
                    print(f"Załadowano model dla miasta {city_pl}")
                except Exception as e:
                    st.error(f"Błąd ładowania modelu dla {city_pl}: {str(e)}")
            else:
                st.error(f"Nie znaleziono modelu: {model_path}")
    
    def predict(self, city: str, features: dict) -> float:
        """
        Przewiduje cenę dla danego miasta i zestawu cech
        
        Args:
            city: nazwa miasta
            features: słownik z cechami mieszkania
            
        Returns:
            float: przewidywana cena
        """
        if city.lower() not in self.models:
            raise ValueError(f"Brak modelu dla miasta {city}")
            
        # Przygotowanie danych wejściowych
        feature_array = np.array([[
            features['squareMeters'],
            features['rooms'],
            features['buildYear'],
            features['centreDistance'],
            features['hasParkingSpace'],
            features['hasBalcony'],
            features['hasElevator'],
            features['hasSecurity'],
            features['hasStorageRoom'],
            features['age_new'],
            features['age_contemporary'],
            features['age_older'],
            features['age_old'],
            features['zone_centrum'],
            features['zone_bliska_strefa'],
            features['zone_srednia_strefa'],
            features['zone_peryferia']
        ]])
        
        return self.models[city.lower()].predict(feature_array)[0]

def create_input_features():
    """Tworzy formularz do wprowadzania cech mieszkania"""
    col1, col2 = st.columns(2)
    
    with col1:
        squareMeters = st.number_input('Powierzchnia (m²)', min_value=20.0, max_value=150.0, value=50.0)
        rooms = st.number_input('Liczba pokoi', min_value=1, max_value=5, value=2)
        buildYear = st.number_input('Rok budowy', min_value=1900, max_value=2024, value=2000)
        centreDistance = st.number_input('Odległość od centrum (km)', min_value=0.0, max_value=15.0, value=5.0)
        
        # Automatyczne określenie strefy i kategorii wieku
        zone = get_zone_from_distance(centreDistance)
        age_category = get_building_age_category(buildYear)
        
        st.info(f'Strefa: {zone}')
        st.info(f'Kategoria wieku budynku: {age_category}')
        
    with col2:
        hasParkingSpace = st.checkbox('Miejsce parkingowe')
        hasBalcony = st.checkbox('Balkon')
        hasElevator = st.checkbox('Winda')
        hasSecurity = st.checkbox('Ochrona')
        hasStorageRoom = st.checkbox('Komórka lokatorska')
    
    # Przygotowanie cech kategorycznych
    age_features = {
        'age_new': 1 if age_category == 'Nowy' else 0,
        'age_contemporary': 1 if age_category == 'Współczesny' else 0,
        'age_older': 1 if age_category == 'Starszy' else 0,
        'age_old': 1 if age_category == 'Stary' else 0
    }
    
    zone_features = {
        'zone_centrum': 1 if zone == 'Centrum' else 0,
        'zone_bliska_strefa': 1 if zone == 'Bliska strefa' else 0,
        'zone_srednia_strefa': 1 if zone == 'Średnia strefa' else 0,
        'zone_peryferia': 1 if zone == 'Peryferia' else 0
    }
    
    features = {
        'squareMeters': squareMeters,
        'rooms': rooms,
        'buildYear': buildYear,
        'centreDistance': centreDistance,
        'hasParkingSpace': int(hasParkingSpace),
        'hasBalcony': int(hasBalcony),
        'hasElevator': int(hasElevator),
        'hasSecurity': int(hasSecurity),
        'hasStorageRoom': int(hasStorageRoom),
        **age_features,
        **zone_features
    }
    
    return features

def show_prediction_tab():
    """Wyświetla zakładkę z predykcją ceny"""
    st.title("🏠 Home Price")
    city = st.selectbox(
        'Wybierz miasto',
        ['Warszawa', 'Kraków', 'Łódź', 'Wrocław', 'Gdańsk']
    )
    
    features = create_input_features()
    
    if st.button('Oblicz cenę'):
        predictor = RealEstatePredictor()
        try:
            predicted_price = predictor.predict(city, features)
            
            st.success(f'Przewidywana cena: {predicted_price:,.2f} PLN')
            
            # Dodatkowe informacje
            st.info("""
            ℹ️ Przewidywana cena jest szacunkowa i może się różnić od rzeczywistej.
            Model uwzględnia historyczne dane z rynku nieruchomości.
            """)
            
            # Wykres wpływu powierzchni na cenę
            st.subheader("Wpływ powierzchni na cenę")
            square_meters_range = np.linspace(20, 150, 50)
            prices = []
            
            for sm in square_meters_range:
                features_copy = features.copy()
                features_copy['squareMeters'] = sm
                price = predictor.predict(city, features_copy)
                prices.append(price)
            
            fig = px.line(
                x=square_meters_range, 
                y=prices,
                labels={'x': 'Powierzchnia (m²)', 'y': 'Przewidywana cena (PLN)'}
            )
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f'Wystąpił błąd: {str(e)}')

def show_models_info_tab():
    """Wyświetla zakładkę z informacjami o modelach"""
    st.title("📊 Informacje o modelach")
    
    metrics = {
        'Warszawa': {'R2': 0.8932, 'MAPE': 8.38, 'MAE': 88720.88},
        'Kraków': {'R2': 0.9057, 'MAPE': 7.87, 'MAE': 74493.46},
        'Łódź': {'R2': 0.8725, 'MAPE': 9.92, 'MAE': 41365.80},
        'Wrocław': {'R2': 0.8384, 'MAPE': 7.11, 'MAE': 54827.45},
        'Gdańsk': {'R2': 0.8348, 'MAPE': 10.3, 'MAE': 88195.84}
    }
    
    # Wykres porównawczy metryk
    metrics_df = pd.DataFrame(metrics).T
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("R² Score")
        fig = px.bar(metrics_df, y='R2')
        st.plotly_chart(fig)
        
    with col2:
        st.subheader("MAPE (%)")
        fig = px.bar(metrics_df, y='MAPE')
        st.plotly_chart(fig)
    
    # Tabela z metrykami
    st.subheader("Szczegółowe metryki")
    st.dataframe(metrics_df)
    
    st.markdown("""
    ### Interpretacja metryk:
    - **R² Score** - współczynnik determinacji, im bliżej 1, tym lepszy model
    - **MAPE** - średni bezwzględny błąd procentowy
    - **MAE** - średni błąd bezwzględny w PLN
    """)

def show_about_tab():
    """Wyświetla zakładkę z informacjami o projekcie"""
    st.title("ℹ️ O projekcie")
    
    st.markdown("""
    ### Projekt predykcji cen nieruchomości
    
    Model został wytrenowany na danych historycznych z rynku nieruchomości z pięciu największych miast w Polsce.
    
    #### Użyte technologie:
    - Python 3.13+
    - scikit-learn (Random Forest Regressor)
    - pandas, numpy
    - Streamlit
    - joblib
    
    #### Zbiór danych:
    - Ponad 50,000 ogłoszeń nieruchomości
    - Okres: 01.2024 - 06.2024
    - 5 miast: Warszawa, Kraków, Łódź, Wrocław, Gdańsk
    
    #### Metodologia:
    1. Wstępne przetwarzanie danych
    2. Utworzenie dedykowanych kategorii (wiek budynku, odległośc od centrum)
    3. Usuwanie duplikatów
    4. Uzupełnienie brakujących danych            
    5. Trening i ewaluacja modelu Random Forest
    6. Walidacja krzyżowa
    
    #### Ograniczenia modelu:
    - Model bazuje na danych historycznych
    - Nie uwzględnia wszystkich czynników wpływających na cenę
    - Może mieć problemy z przewidywaniem cen dla nietypowych nieruchomości
    """)

def main():
    """Główna funkcja aplikacji"""
    tabs = st.tabs(["Predykcja ceny", "Informacje o modelach", "O projekcie"])
    
    with tabs[0]:
        show_prediction_tab()
        
    with tabs[1]:
        show_models_info_tab()
        
    with tabs[2]:
        show_about_tab()

if __name__ == "__main__":
    main()