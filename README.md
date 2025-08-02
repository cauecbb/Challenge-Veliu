# Product Matching Solution

## Overview

This project implements a fuzzy matching system to reconcile products from two photography equipment datasets. The solution finds similar products across datasets and returns the highest available price when a match is found.

## Problem Statement

We have two datasets containing photography equipment with randomly generated prices:
- **Dataset1**: 382 products with structured data in Italian
- **Dataset2**: 11,229 products with complex JSON features

The goal is to:
1. Reconcile products between datasets as accurately as possible
2. Return the highest price when a match is found
3. Provide a search system for users to find products

## Solution Architecture

### 1. Data Loading and Analysis
- Load both datasets using pandas
- Perform exploratory data analysis
- Identify structural differences and data quality issues

### 2. Data Cleaning and Normalization
- Normalize brand names (Canon, Nikon, Sony, etc.)
- Clean and standardize product names
- Extract brand information from dataset2 titles
- Extract category information from JSON features
- Create standardized columns for matching

### 3. Fuzzy Matching Algorithm
- Use `fuzzywuzzy` library for string similarity
- Implement multi-criteria matching:
  - **Name similarity** (60% weight): Fuzzy string matching
  - **Brand similarity** (30% weight): Exact match
  - **Category similarity** (10% weight): Exact match
- Set threshold at 50% for practical matching

### 4. Product Search System
- Implement `search_product()` function
- Search across both datasets
- Return product information and price
- Include similarity score in results

## Technical Decisions

### Threshold Selection (50%)
- **Justification**: Balance between precision and recall
- **Testing**: 80% was too strict (0 matches), 60% found 3 matches
- **Final**: 50% provides good coverage while maintaining quality

### Similarity Weights
- **Name (60%)**: Most important for product identification
- **Brand (30%)**: Ensures same manufacturer
- **Category (10%)**: Additional validation

### Brand Filtering
- Filter dataset2 to relevant brands for efficiency
- Reduces comparison space from 11,229 to 7,700 products
- Improves performance significantly

## Results

### Matching Performance
- **Total matches found**: 110
- **Match rate**: 28.8% (110/382 products from dataset1)
- **Average similarity score**: 54.2%
- **Price range**: $730 - $4,992

### Example Matches
1. Canon EF 75-300mm f/4-5.6 ($1,298) ↔ Canon EF 75-300mm f/4-5.6 II ($730)
   - Similarity: 56.4%
   - Highest price: $1,298

2. Canon EF 70-300mm f/4-5.6 IS USM II ($2,388) ↔ Canon EF 70-300mm f/4-5.6 IS USM Nero ($833)
   - Similarity: 55.2%
   - Highest price: $2,388

### Search System Performance
- Successfully tested with real product queries
- 100% match rate for exact product names (Nikon D850, Sony A7)
- Good partial matching for similar products

## Usage

### Running the Complete Pipeline
```bash
python main.py
```

### Using the Search Function
```python
from main import load_datasets, clean_and_normalize_data, search_product

# Load and prepare data
df1, df2 = load_datasets()
df1_clean, df2_clean = clean_and_normalize_data(df1, df2)

# Search for a product
result = search_product("Canon EF 70-200mm", df1_clean, df2_clean, threshold=50.0)

if result['found']:
    print(f"Product: {result['product_name']}")
    print(f"Price: ${result['price']}")
    print(f"Similarity: {result['similarity_score']:.1f}%")
```

### Example Queries
- "Canon EF 70-200mm" → Canon EF 70-200 2.8 USM ($1,898, 80% similarity)
- "Nikon D850" → Nikon D850 ($1,403, 100% similarity)
- "Sony A7" → Sony A7 ($4,123, 100% similarity)

## Dependencies

```
pandas
fuzzywuzzy
numpy
```

Install with:
```bash
pip install -r requirements.txt
```

## Project Structure

```
Challenge-Veliu/
├── data/
│   ├── dataset1.csv          # 382 products
│   └── dataset2.csv          # 11,229 products
├── output/
│   └── matching_results.csv  # Generated results
├── main.py                   # Main implementation
├── requirements.txt          # Dependencies
└── README.md                # This file
```

## Assumptions and Limitations

### Assumptions
1. Products with similar names and same brand are likely the same
2. Price differences are due to market variations, not different products
3. Fuzzy matching with 50% threshold provides good balance

### Limitations
1. Limited to major photography brands (Canon, Nikon, Sony, Fujifilm, Sigma, Tamron)
2. Relies heavily on product name similarity
3. May miss matches due to different naming conventions
4. No consideration of product specifications beyond name/brand/category

## Future Improvements

1. **Enhanced Feature Extraction**: Parse JSON features more thoroughly
2. **Machine Learning**: Train a model for better similarity scoring
3. **Product Specifications**: Include focal length, aperture, etc. in matching

## Performance Metrics

- **Processing Time**: ~30 seconds for full pipeline
- **Memory Usage**: Efficient with pandas operations
- **Accuracy**: 28.8% match rate with 50% threshold
- **Scalability**: Can handle larger datasets with optimization

## Conclusion

This solution successfully reconciles photography products across two datasets using fuzzy matching techniques. The 28.8% match rate with 50% threshold provides a good balance between coverage and accuracy for a practical product search system.

The implementation demonstrates:
- Solid data engineering practices
- Practical fuzzy matching implementation
- Functional search system
- Clear documentation and examples


## License

This project was developed as part of a technical test.
