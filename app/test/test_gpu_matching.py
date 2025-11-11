"""
Test GPU fuzzy matching with actual data
"""
import pandas as pd
from app.data_utils import clean_laptops_df, fuzzy_match_gpu

# Load actual data
print("Loading laptop data...")
laptop_df = pd.read_csv('csv/laptop_data.csv')
print(f"Loaded {len(laptop_df)} laptops\n")

# Load GPU database
print("Loading GPU database...")
gpu_df = pd.read_csv('csv/gpu_data.csv')
print(f"Loaded {len(gpu_df)} GPU entries\n")

# Show sample GPU entries from laptop data
print("=== Sample GPU names from laptop_data.csv ===")
sample_gpus = laptop_df['Gpu'].head(15).tolist()
for i, gpu in enumerate(sample_gpus, 1):
    print(f"{i}. {gpu}")

print("\n=== Testing fuzzy matching ===")
# Clean GPU names and prepare list
gpu_df['GPU Mark'] = gpu_df['GPU Mark'].astype(str).str.replace(',', '').str.strip()
gpu_list = gpu_df['Nama GPU'].dropna().tolist()

print(f"GPU database has {len(gpu_list)} entries\n")

# Test matching for sample GPUs
for gpu in sample_gpus[:8]:
    matched = fuzzy_match_gpu(str(gpu), gpu_list, threshold=60)
    if matched:
        score = gpu_df[gpu_df['Nama GPU'] == matched]['GPU Mark'].values
        score_val = score[0] if len(score) > 0 else 'N/A'
        print(f"Input:   {gpu}")
        print(f"Match:   {matched}")
        print(f"Score:   {score_val}")
        print("-" * 70)
    else:
        print(f"Input:   {gpu}")
        print(f"Match:   NO MATCH FOUND")
        print("-" * 70)

print("\n=== Running clean_laptops_df on sample ===")
# Test with small sample
test_df = laptop_df.head(30).copy()
cleaned_df = clean_laptops_df(test_df)

# Show results
print("\nResults:")
print(cleaned_df[['Gpu', 'GPU_Score']].to_string())

# Count how many got matched
matched_count = cleaned_df['GPU_Score'].notna().sum()
total_count = len(cleaned_df)
print(f"\nMatched: {matched_count}/{total_count} ({matched_count/total_count*100:.1f}%)")

# Show which GPUs didn't match
unmatched = cleaned_df[cleaned_df['GPU_Score'].isna()]
if not unmatched.empty:
    print("\n=== Unmatched GPUs ===")
    for gpu in unmatched['Gpu'].unique():
        print(f"- {gpu}")
