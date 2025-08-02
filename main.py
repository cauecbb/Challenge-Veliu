import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple
from fuzzywuzzy import fuzz
import re

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

def calculate_similarity_score(product1: Dict, product2: Dict) -> float:
    """
    Calculate similarity score between two products using multiple criteria.
    
    Args:
        product1: Product from dataset1
        product2: Product from dataset2
        
    Returns:
        float: Similarity score (0-100)
    """
    # Initialize weights for different matching criteria
    name_weight = 0.6
    brand_weight = 0.3
    category_weight = 0.1
    
    # 1. Name similarity (using fuzzy string matching)
    name_similarity = fuzz.ratio(product1['name_clean'], product2['name_clean'])
    
    # 2. Brand similarity (exact match)
    brand_similarity = 100 if product1['brand_normalized'] == product2['brand_normalized'] else 0
    
    # 3. Category similarity (exact match)
    category_similarity = 100 if product1['category_normalized'] == product2['category_extracted'] else 0
    
    # Calculate score average
    total_score = (
        name_similarity * name_weight +
        brand_similarity * brand_weight +
        category_similarity * category_weight
    )
    
    return total_score

def find_matches(df1_clean: pd.DataFrame, df2_clean: pd.DataFrame, threshold: float = 50.0) -> List[Dict]:
    """
    Find matches between products from both datasets.
    
    Args:
        df1_clean: Cleaned dataset1
        df2_clean: Cleaned dataset2
        threshold: Minimum similarity score to consider a match
        
    Returns:
        List[Dict]: List of matches with details
    """
    print(f"\n=== FUZZY MATCHING IMPLEMENTATION ===")
    print(f"Finding matches with threshold: {threshold}")
    
    matches = []
    
    # Filter dataset2 to only include relevant brands for efficiency
    relevant_brands = ['canon', 'nikon', 'sony', 'fujifilm', 'sigma', 'tamron']
    df2_filtered = df2_clean[df2_clean['brand_normalized'].isin(relevant_brands)].copy()
    
    print(f"Comparing {len(df1_clean)} products from dataset1 with {len(df2_filtered)} relevant products from dataset2")
    
    # For each product in dataset1, find best match in dataset2
    for idx1, product1 in df1_clean.iterrows():
        best_match = None
        best_score = 0
        
        # Convert product1 to dict for easier access
        product1_dict = product1.to_dict()
        
        for idx2, product2 in df2_filtered.iterrows():
            product2_dict = product2.to_dict()
            
            # Calculate similarity score
            score = calculate_similarity_score(product1_dict, product2_dict)
            
            # Update best match if score is higher and above threshold
            if score > best_score and score >= threshold:
                best_score = score
                best_match = {
                    'dataset1_idx': idx1,
                    'dataset2_idx': idx2,
                    'product1_name': product1['Nome'],
                    'product2_name': product2['Title'],
                    'product1_brand': product1['brand_normalized'],
                    'product2_brand': product2['brand_normalized'],
                    'product1_price': product1['price'],
                    'product2_price': product2['price'],
                    'similarity_score': score,
                    'highest_price': max(product1['price'], product2['price'])
                }
        
        if best_match:
            matches.append(best_match)
    
    print(f"Found {len(matches)} matches above threshold {threshold}")
    
    # Show some examples
    if matches:
        print(f"\nExample matches:")
        for i, match in enumerate(matches[:3]):
            print(f"Match {i+1}:")
            print(f"  Product1: {match['product1_name']} (${match['product1_price']})")
            print(f"  Product2: {match['product2_name']} (${match['product2_price']})")
            print(f"  Similarity: {match['similarity_score']:.1f}%")
            print(f"  Highest price: ${match['highest_price']}")
            print()
    
    return matches

def search_product(product_name: str, df1_clean: pd.DataFrame, df2_clean: pd.DataFrame, threshold: float = 50.0) -> Dict:
    """
    Search for a product and return the highest price if a match is found.
    
    Args:
        product_name: Name of the product to search for
        df1_clean: Cleaned dataset1
        df2_clean: Cleaned dataset2
        threshold: Minimum similarity score to consider a match
        
    Returns:
        Dict: Search result with product info and highest price
    """
    print(f"\n=== SEARCHING FOR: {product_name} ===")
    
    # Clean the search query
    search_query = product_name.lower()
    search_query = re.sub(r'[^\w\s]', ' ', search_query)
    search_query = re.sub(r'\s+', ' ', search_query).strip()
    
    best_match = None
    best_score = 0
    
    # Search in dataset1 first
    for idx, product in df1_clean.iterrows():
        score = fuzz.ratio(search_query, product['name_clean'])
        if score > best_score:
            best_score = score
            best_match = {
                'dataset': 'dataset1',
                'product_name': product['Nome'],
                'brand': product['brand_normalized'],
                'price': product['price'],
                'similarity_score': score
            }
    
    # Search in dataset2
    relevant_brands = ['canon', 'nikon', 'sony', 'fujifilm', 'sigma', 'tamron']
    df2_filtered = df2_clean[df2_clean['brand_normalized'].isin(relevant_brands)]
    
    for idx, product in df2_filtered.iterrows():
        score = fuzz.ratio(search_query, product['name_clean'])
        if score > best_score:
            best_score = score
            best_match = {
                'dataset': 'dataset2',
                'product_name': product['Title'],
                'brand': product['brand_normalized'],
                'price': product['price'],
                'similarity_score': score
            }
    
    # Check if we found a match above threshold
    if best_match and best_match['similarity_score'] >= threshold:
        print(f"✅ MATCH FOUND!")
        print(f"Product: {best_match['product_name']}")
        print(f"Brand: {best_match['brand']}")
        print(f"Price: ${best_match['price']}")
        print(f"Similarity: {best_match['similarity_score']:.1f}%")
        print(f"Dataset: {best_match['dataset']}")
        
        return {
            'found': True,
            'product_name': best_match['product_name'],
            'brand': best_match['brand'],
            'price': best_match['price'],
            'similarity_score': best_match['similarity_score'],
            'dataset': best_match['dataset']
        }
    else:
        print(f"❌ No match found above threshold {threshold}")
        if best_match:
            print(f"Best match was: {best_match['product_name']} (similarity: {best_match['similarity_score']:.1f}%)")
        
        return {
            'found': False,
            'message': f"No match found above threshold {threshold}",
            'best_match': best_match
        }

def test_search_system(df1_clean: pd.DataFrame, df2_clean: pd.DataFrame) -> None:
    """
    Test the search system with example queries.
    
    Args:
        df1_clean: Cleaned dataset1
        df2_clean: Cleaned dataset2
    """
    print(f"\n=== TESTING SEARCH SYSTEM ===")
    
    # Test queries
    test_queries = [
        "Canon EF 70-200mm",
        "Nikon D850",
        "Sony A7",
        "Sigma 35mm",
        "Non-existent product"
    ]
    
    for query in test_queries:
        result = search_product(query, df1_clean, df2_clean, threshold=50.0)
        print("-" * 50)

def analyze_results(matches: List[Dict], df1_clean: pd.DataFrame, df2_clean: pd.DataFrame) -> None:
    """
    Analyze and summarize the matching results.
    
    Args:
        matches: List of matches found
        df1_clean: Cleaned dataset1
        df2_clean: Cleaned dataset2
    """
    print(f"\n=== RESULTS ANALYSIS ===")
    
    # Basic statistics
    print(f"Total matches found: {len(matches)}")
    print(f"Dataset1 products: {len(df1_clean)}")
    print(f"Dataset2 products: {len(df2_clean)}")
    print(f"Match rate: {len(matches)/len(df1_clean)*100:.1f}%")
    
    # Price analysis
    if matches:
        prices = [match['highest_price'] for match in matches]
        print(f"\nPrice Statistics for Matches:")
        print(f"Average highest price: ${np.mean(prices):.2f}")
        print(f"Min highest price: ${min(prices)}")
        print(f"Max highest price: ${max(prices)}")
        
        # Similarity score analysis
        scores = [match['similarity_score'] for match in matches]
        print(f"\nSimilarity Score Statistics:")
        print(f"Average similarity: {np.mean(scores):.1f}%")
        print(f"Min similarity: {min(scores):.1f}%")
        print(f"Max similarity: {max(scores):.1f}%")
        
        # Brand analysis
        brands = [match['product1_brand'] for match in matches]
        brand_counts = pd.Series(brands).value_counts()
        print(f"\nMatches by Brand:")
        for brand, count in brand_counts.head().items():
            print(f"  {brand}: {count} matches")
    
    # Save results to file
    if matches:
        results_df = pd.DataFrame(matches)
        results_df.to_csv('output/matching_results.csv', index=False)
        print(f"\nResults saved to: output/matching_results.csv")

def main():
    """Main function to execute the pipeline."""
    print("=== PRODUCT MATCHING SOLUTION ===")
    
    # Load data
    df1, df2 = load_datasets()
    
    # Analysis
    analyze_datasets(df1, df2)
    
    # Clean and normalize data
    df1_clean, df2_clean = clean_and_normalize_data(df1, df2)
    
    # Find matches using fuzzy matching
    matches = find_matches(df1_clean, df2_clean, threshold=50.0)
    
    # Test search system
    test_search_system(df1_clean, df2_clean)
    
    # Analyze results and generate summary
    analyze_results(matches, df1_clean, df2_clean)

    print("\nCheck README.md for documentation and usage examples\n)
 


if __name__ == "__main__":
    main() 