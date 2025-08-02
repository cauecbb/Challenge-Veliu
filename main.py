import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple

def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the two datasets.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Dataset1 and Dataset2
    """
    print("Loading datasets...")
    
    # Load dataset1
    df1 = pd.read_csv('data/dataset1.csv')
    print(f"Dataset1 loaded: {len(df1)} products")
    
    # Load dataset2
    df2 = pd.read_csv('data/dataset2.csv')
    print(f"Dataset2 loaded: {len(df2)} products")
    
    return df1, df2

def analyze_datasets(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    """
    Basic analysis of the datasets.
    
    Args:
        df1: Dataset1
        df2: Dataset2
    """
    print("\n=== ANALYSIS ===")
    
    # Dataset1
    print(f"\nDataset1:")
    print(f"- Shape: {df1.shape}")
    print(f"- Columns: {list(df1.columns)}")
    print(f"- Data types:")
    for col in df1.columns:
        print(f"  {col}: {df1[col].dtype}")
    
    # Dataset2
    print(f"\nDataset2:")
    print(f"- Shape: {df2.shape}")
    print(f"- Columns: {list(df2.columns)}")
    print(f"- Data types:")
    for col in df2.columns:
        print(f"  {col}: {df2[col].dtype}")
    
    # Price analysis
    print(f"\nPrice Analysis:")
    print(f"Dataset1 - Average price: {df1['price'].mean():.2f}")
    print(f"Dataset1 - Min price: {df1['price'].min()}")
    print(f"Dataset1 - Max price: {df1['price'].max()}")
    print(f"Dataset2 - Average price: {df2['price'].mean():.2f}")
    print(f"Dataset2 - Min price: {df2['price'].min()}")
    print(f"Dataset2 - Max price: {df2['price'].max()}")
    
    # Data samples
    print(f"\nDataset1 Samples:")
    print(df1.head(3)[['Nome', 'Marchio', 'Categoria', 'price']])
    
    print(f"\nDataset2 Samples:")
    print(df2.head(3)[['Title', 'IntName', 'Category_Name', 'price']])

def clean_and_normalize_data(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean and normalize data from both datasets for matching.
    
    Args:
        df1: Dataset1
        df2: Dataset2
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Cleaned datasets
    """
    print("\n=== DATA CLEANING AND NORMALIZATION ===")
    
    # Create copies to avoid modifying original data
    df1_clean = df1.copy()
    df2_clean = df2.copy()
    
    # Normalize brand names in dataset1
    print("Normalizing brand names in dataset1...")
    brand_mapping = {
        'Canon & compatible': 'Canon',
        'Nikon & compatible': 'Nikon',
        'Sony': 'Sony',
        'Fujifilm': 'Fujifilm',
        'Sigma': 'Sigma',
        'Tamron': 'Tamron'
    }
    df1_clean['brand_normalized'] = df1_clean['Marchio'].map(brand_mapping)
    
    # Clean product names in dataset1
    print("Cleaning product names in dataset1...")
    df1_clean['name_clean'] = df1_clean['Nome'].str.lower()
    df1_clean['name_clean'] = df1_clean['name_clean'].str.replace(r'[^\w\s]', ' ', regex=True)
    df1_clean['name_clean'] = df1_clean['name_clean'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Extract brand from dataset2
    print("Extracting brand information from dataset2...")
    def extract_brand(title, intname):
        """Extract brand from title or internal name"""
        text = str(title) + ' ' + str(intname)
        text = text.lower()

        known_brands = ['canon', 'nikon', 'sony', 'fujifilm', 'sigma', 'tamron']
        for brand in known_brands:
            if brand in text:
                return brand
        return 'Other'

    df2_clean['brand_normalized'] = df2_clean.apply(
        lambda x: extract_brand(x['Title'], x['IntName']), axis=1
    )
    
    # Clean product names in dataset2
    print("Cleaning product names in dataset2...")
    df2_clean['name_clean'] = df2_clean['Title'].str.lower()
    df2_clean['name_clean'] = df2_clean['name_clean'].str.replace(r'[^\w\s]', ' ', regex=True)
    df2_clean['name_clean'] = df2_clean['name_clean'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Extract category information from dataset2
    print("Extracting category information from dataset2...")
    def extract_category_from_json(features_str):
        """Extract category information from JSON features"""
        try:
            if pd.isna(features_str):
                return 'Unknown'
            
            # Simple extraction - look for common photography terms
            features_lower = str(features_str).lower()
            
            if 'lens' in features_lower or 'obiettivo' in features_lower:
                return 'Lens'
            elif 'camera' in features_lower or 'fotocamera' in features_lower:
                return 'Camera'
            elif 'flash' in features_lower:
                return 'Flash'
            elif 'tripod' in features_lower:
                return 'Tripod'
            else:
                return 'Other'
        except:
            return 'Unknown'
    
    df2_clean['category_extracted'] = df2_clean['features'].apply(extract_category_from_json)
    
    # Create standardized category mapping
    print("Creating standardized category mapping...")
    category_mapping = {
        'Obiettivi': 'Lens',
        'Reflex': 'Camera',
        'Mirrorless': 'Camera'
    }
    df1_clean['category_normalized'] = df1_clean['Categoria'].map(category_mapping).fillna('Other')
    
    print(f"Data cleaning completed!")
    print(f"Dataset1: {len(df1_clean)} products")
    print(f"Dataset2: {len(df2_clean)} products")
    
    return df1_clean, df2_clean

def main():
    """Main function to execute the pipeline (to be implemented)"""
    print("=== PRODUCT MATCHING SOLUTION ===")
    
    # Load data
    df1, df2 = load_datasets()
    
    # Analysis
    analyze_datasets(df1, df2)
    
    # Clean and normalize data
    df1_clean, df2_clean = clean_and_normalize_data(df1, df2)
    

if __name__ == "__main__":
    main() 