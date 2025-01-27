import pandas as pd
import numpy as np
from typing import List, Dict
import os
from datetime import datetime

def categorize_building_by_year(year: float, current_year: int = 2025) -> str:
    """Categorize building based on construction year"""
    age = current_year - year
    if age <= 5:
        return "new"
    elif age <= 15:
        return "contemporary"
    elif age <= 30:
        return "older"
    else:
        return "old"

def categorize_distance(distance: float) -> str:
    """Categorize distance to city center into zones"""
    if distance <= 2:
        return "centrum"
    elif distance <= 5:
        return "bliska_strefa"
    elif distance <= 8:
        return "srednia_strefa"
    else:
        return "peryferia"

def load_and_merge_data(data_dir: str) -> pd.DataFrame:
    """Load and merge all CSV files from the specified directory."""
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files")
    
    dfs = []
    for file in csv_files:
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)
        print(f"Reading {file}: {len(df)} records")
        dfs.append(df)
    
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal records after merge: {len(merged_df)}")
    
    duplicates = merged_df.duplicated(subset=['id', 'price', 'squareMeters', 'rooms'], keep='first')
    print(f"Found {duplicates.sum()} duplicate records")
    
    return merged_df

def clean_and_prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare data for the real estate price prediction model."""
    print("\nStarting data cleaning process...")
    
    # Keep initial columns including 'floor' for elevator logic
    initial_columns = [
        'id', 'city', 'squareMeters', 'rooms', 'buildYear', 'centreDistance',
        'hasParkingSpace', 'hasBalcony', 'hasElevator', 'hasSecurity',
        'hasStorageRoom', 'price', 'floor'
    ]
    
    df = df[initial_columns].copy()
    print(f"Selected {len(initial_columns)} columns")
    
    # Remove duplicates
    initial_len = len(df)
    df = df.drop_duplicates(
        subset=['id', 'price', 'squareMeters', 'rooms'], 
        keep='first'
    )
    removed_duplicates = initial_len - len(df)
    print(f"Removed {removed_duplicates} duplicate records")
    
    # Keep only top 5 cities by number of listings
    city_counts = df['city'].value_counts()
    top_5_cities = city_counts.head(5).index.tolist()
    print(f"\nTop 5 cities by number of listings:")
    for city in top_5_cities:
        print(f"{city}: {city_counts[city]} listings")
    
    df = df[df['city'].isin(top_5_cities)]
    print(f"Kept {len(df)} records from top 5 cities")
    
    # Convert boolean columns from 'yes'/'no' to TRUE/FALSE
    boolean_columns = [
        'hasParkingSpace', 'hasBalcony', 'hasSecurity', 'hasStorageRoom'
    ]
    
    for col in boolean_columns:
        df[col] = df[col].map({'yes': True, 'no': False})
    
    # Handle hasElevator with floor logic
    print("\nHandling hasElevator values:")
    df['hasElevator'] = df['hasElevator'].map({'yes': True, 'no': False})
    
    # Fill NaN values based on floor number
    df['floor'] = pd.to_numeric(df['floor'], errors='coerce')
    missing_elevator_mask = df['hasElevator'].isna()
    high_floor_mask = df['floor'] >= 5
    
    df.loc[missing_elevator_mask & high_floor_mask, 'hasElevator'] = True
    df.loc[missing_elevator_mask & ~high_floor_mask, 'hasElevator'] = False
    
    # Handle missing buildYear values
    missing_build_year = df['buildYear'].isna().sum()
    city_medians = df.groupby('city')['buildYear'].median()
    df['buildYear'] = df.apply(
        lambda x: city_medians[x['city']] if pd.isna(x['buildYear']) else x['buildYear'],
        axis=1
    )
    print(f"\nFilled {missing_build_year} missing buildYear values with city medians")
    
    # Add building age categories and distance zones
    df['buildingAgeCategory'] = df['buildYear'].apply(categorize_building_by_year)
    df['locationZone'] = df['centreDistance'].apply(categorize_distance)
    
    # Remove rows with missing values in important columns
    initial_len = len(df)
    df = df.dropna(subset=['squareMeters', 'rooms', 'centreDistance', 'price'])
    removed_missing = initial_len - len(df)
    print(f"Removed {removed_missing} rows with missing values")
    
    # Convert categories to dummy variables
    df = pd.get_dummies(df, columns=['city', 'buildingAgeCategory', 'locationZone'], 
                       prefix=['city', 'age', 'zone'])
    
    # Remove temporary columns
    df = df.drop(['floor'], axis=1)
    
    # Ensure final column order
    final_columns = [
        'id', 'squareMeters', 'rooms', 'buildYear',
        'centreDistance', 'hasParkingSpace', 'hasBalcony', 
        'hasElevator', 'hasSecurity', 'hasStorageRoom', 'price',
        'city_gdansk', 'city_krakow', 'city_lodz', 'city_warszawa', 'city_wroclaw',
        'age_new', 'age_contemporary', 'age_older', 'age_old',
        'zone_centrum', 'zone_bliska_strefa', 'zone_srednia_strefa', 'zone_peryferia'
    ]
    
    df = df[final_columns]
    return df

def analyze_data(df: pd.DataFrame) -> None:
    """Print detailed analysis of the processed data."""
    print("\nDataset Analysis:")
    print(f"Total records: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    
    print("\nNumerical features statistics:")
    numerical_cols = ['squareMeters', 'rooms', 'buildYear', 'centreDistance', 'price']
    print(df[numerical_cols].describe())
    
    print("\nBoolean features statistics:")
    bool_cols = ['hasParkingSpace', 'hasBalcony', 'hasElevator', 
                 'hasSecurity', 'hasStorageRoom']
    for col in bool_cols:
        true_count = df[col].sum()
        false_count = len(df) - true_count
        print(f"\n{col}:")
        print(f"TRUE: {true_count} ({(true_count/len(df))*100:.2f}%)")
        print(f"FALSE: {false_count} ({(false_count/len(df))*100:.2f}%)")

    print("\nBuilding Age Categories distribution:")
    age_cols = [col for col in df.columns if col.startswith('age_')]
    for col in age_cols:
        count = df[col].sum()
        print(f"{col}: {count} ({(count/len(df))*100:.2f}%)")

    print("\nLocation Zones distribution:")
    zone_cols = [col for col in df.columns if col.startswith('zone_')]
    for col in zone_cols:
        count = df[col].sum()
        print(f"{col}: {count} ({(count/len(df))*100:.2f}%)")

    print("\nDistribution of properties by city:")
    city_cols = [col for col in df.columns if col.startswith('city_')]
    for col in city_cols:
        count = df[col].sum()
        print(f"{col[5:]}: {count} ({(count/len(df))*100:.2f}%)")

def main():
    # Set the data directory
    data_dir = '/Users/damian/Documents/PJATK/Przedmioty/2025/SUML/SUML Projekt/dataset'
    output_path = os.path.join(data_dir, 'processed_apartments_data.csv')
    
    # Load and merge data
    print("Loading and merging data...")
    merged_df = load_and_merge_data(data_dir)
    
    # Clean and prepare data
    cleaned_df = clean_and_prepare_data(merged_df)
    
    # Analyze processed data
    analyze_data(cleaned_df)
    
    # Save processed data
    cleaned_df.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to: {output_path}")

if __name__ == "__main__":
    main()