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

def main():
    """Main function to execute the pipeline (to be implemented)"""
    print("=== PRODUCT MATCHING SOLUTION ===")
    
    # Load data
    df1, df2 = load_datasets()
    
    # Analysis
    analyze_datasets(df1, df2)

    # to be implemented

if __name__ == "__main__":
    main() 