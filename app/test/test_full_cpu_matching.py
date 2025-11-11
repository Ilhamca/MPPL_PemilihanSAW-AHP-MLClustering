"""
Full test of CPU fuzzy matching on entire dataset
"""
import pandas as pd
from app.data_utils import clean_laptops_df

# Load actual data
print("Loading laptop data...")
laptop_df = pd.read_csv('csv/laptop_data.csv')
print(f"Loaded {len(laptop_df)} laptops\n")

print("=== Running clean_laptops_df on full dataset ===")
print("This may take a moment...\n")

# Clean the full dataset
cleaned_df = clean_laptops_df(laptop_df)

# Show results
matched_count = cleaned_df['Prosesor_Score'].notna().sum()
total_count = len(cleaned_df)
match_rate = (matched_count/total_count*100)

print(f"✅ Matched: {matched_count}/{total_count} ({match_rate:.1f}%)")

# Show which CPUs didn't match
unmatched = cleaned_df[cleaned_df['Prosesor_Score'].isna()]
if not unmatched.empty:
    print(f"\n❌ Unmatched: {len(unmatched)} CPUs")
    print("\n=== Unmatched CPU Names ===")
    unique_unmatched = unmatched['Cpu'].value_counts()
    for cpu, count in unique_unmatched.items():
        print(f"  {cpu} (appears {count}x)")
else:
    print("\n🎉 All CPUs matched successfully!")

# Show sample of matched CPUs with scores
print("\n=== Sample Matched CPUs ===")
matched_sample = cleaned_df[cleaned_df['Prosesor_Score'].notna()][['Cpu', 'Prosesor_Score']].head(15)
print(matched_sample.to_string(index=False))

# Show distribution statistics
print("\n=== CPU Mark Score Statistics ===")
print(cleaned_df['Prosesor_Score'].describe())

# Save to CSV for inspection
output_file = 'test_output_with_cpu_scores.csv'
cleaned_df.to_csv(output_file, index=False)
print(f"\n✅ Saved results to: {output_file}")
