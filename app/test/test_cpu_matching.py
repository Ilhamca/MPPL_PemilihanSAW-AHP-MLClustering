"""
Test CPU fuzzy matching with actual data
"""
import pandas as pd
from app.data_utils import clean_laptops_df, fuzzy_match_cpu

# Load actual data
print("Loading laptop data...")
laptop_df = pd.read_csv('csv/laptop_data.csv')
print(f"Loaded {len(laptop_df)} laptops\n")

# Load CPU database
print("Loading CPU database...")
cpu_df = pd.read_csv('csv/cpu_data.csv')
print(f"Loaded {len(cpu_df)} CPU entries\n")

# Show sample CPU entries from laptop data
print("=== Sample CPU names from laptop_data.csv ===")
sample_cpus = laptop_df['Cpu'].head(10).tolist()
for i, cpu in enumerate(sample_cpus, 1):
    print(f"{i}. {cpu}")

print("\n=== Testing fuzzy matching ===")
# Clean CPU names and prepare list
cpu_df['CPU Mark'] = cpu_df['CPU Mark'].astype(str).str.replace(',', '').str.strip()
cpu_list = cpu_df['Nama CPU'].dropna().tolist()

print(f"CPU database has {len(cpu_list)} entries\n")

# Test matching for sample CPUs
for cpu in sample_cpus[:5]:
    matched = fuzzy_match_cpu(str(cpu), cpu_list, threshold=60)
    if matched:
        score = cpu_df[cpu_df['Nama CPU'] == matched]['CPU Mark'].values
        score_val = score[0] if len(score) > 0 else 'N/A'
        print(f"Input:   {cpu}")
        print(f"Match:   {matched}")
        print(f"Score:   {score_val}")
        print("-" * 70)
    else:
        print(f"Input:   {cpu}")
        print(f"Match:   NO MATCH FOUND")
        print("-" * 70)

print("\n=== Running full clean_laptops_df ===")
# Test with small sample
test_df = laptop_df.head(20).copy()
cleaned_df = clean_laptops_df(test_df)

# Show results
print("\nResults:")
print(cleaned_df[['Cpu', 'Prosesor_Score']].to_string())

# Count how many got matched
matched_count = cleaned_df['Prosesor_Score'].notna().sum()
total_count = len(cleaned_df)
print(f"\nMatched: {matched_count}/{total_count} ({matched_count/total_count*100:.1f}%)")

# Show which CPUs didn't match
unmatched = cleaned_df[cleaned_df['Prosesor_Score'].isna()]
if not unmatched.empty:
    print("\n=== Unmatched CPUs ===")
    for cpu in unmatched['Cpu'].unique():
        print(f"- {cpu}")
