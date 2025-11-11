"""
Test script to demonstrate fuzzy CPU matching functionality
"""
import pandas as pd
from app.data_utils import fuzzy_match_cpu, clean_laptops_df

# Example CPU names that might not match exactly
test_cpus = [
    "Intel Core i5 2.3GHz",
    "Intel Core i7-8550U",
    "AMD Ryzen 5 3600",
    "Intel i5-10210U @ 1.60GHz",
    "AMD A9-Series A9-9420 3GHz"
]

# Simulated CPU database (you would load this from csv/cpu_data.csv)
cpu_database = [
    "Intel Core i5-2300 @ 2.80GHz",
    "Intel Core i7-8550U @ 1.80GHz",
    "AMD Ryzen 5 3600 6-Core",
    "Intel Core i5-10210U @ 1.60GHz",
    "AMD A9-9420 APU"
]

print("=== Fuzzy CPU Matching Test ===\n")

for cpu in test_cpus:
    match = fuzzy_match_cpu(cpu, cpu_database, threshold=70)
    print(f"Input:   {cpu}")
    print(f"Matched: {match if match else 'No match found'}")
    print("-" * 60)

# Test with actual DataFrame
print("\n=== Testing with DataFrame ===\n")

test_df = pd.DataFrame({
    'Nama': ['Laptop 1', 'Laptop 2', 'Laptop 3'],
    'Cpu': [
        'Intel Core i5 8250U',
        'AMD Ryzen 7 3700U', 
        'Intel Core i7-10510U'
    ],
    'Company': ['Dell', 'HP', 'Lenovo'],
    'TypeName': ['Notebook', 'Notebook', 'Ultrabook'],
    'Ram': ['8GB', '16GB', '16GB'],
    'Memory': ['256GB SSD', '512GB SSD', '1TB SSD'],
    'Weight': ['1.8kg', '1.5kg', '1.2kg']
})

print("Original DataFrame:")
print(test_df[['Nama', 'Cpu']])
print("\nNote: To see actual CPU matching, you need:")
print("1. Install rapidfuzz: pip install rapidfuzz")
print("2. Have cpu_data.csv or cpu_benchmark_scores.csv in csv/ folder")
print("\nThe fuzzy_match_cpu function will find the best match even if:")
print("- CPU names have slight spelling differences")
print("- Extra information is present (like clock speeds)")
print("- Word order is different")
print("- There are abbreviations or missing words")
