"""
Full test of GPU fuzzy matching on entire dataset
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

# Show CPU results
cpu_matched_count = cleaned_df['Prosesor_Score'].notna().sum()
cpu_total_count = len(cleaned_df)
cpu_match_rate = (cpu_matched_count/cpu_total_count*100)

print(f"✅ CPU Matched: {cpu_matched_count}/{cpu_total_count} ({cpu_match_rate:.1f}%)")

# Show GPU results
gpu_matched_count = cleaned_df['GPU_Score'].notna().sum()
gpu_total_count = len(cleaned_df)
gpu_match_rate = (gpu_matched_count/gpu_total_count*100)

print(f"✅ GPU Matched: {gpu_matched_count}/{gpu_total_count} ({gpu_match_rate:.1f}%)")

# Show which GPUs didn't match
unmatched = cleaned_df[cleaned_df['GPU_Score'].isna()]
if not unmatched.empty:
    print(f"\n❌ Unmatched: {len(unmatched)} GPUs")
    print("\n=== Unmatched GPU Names ===")
    unique_unmatched = unmatched['Gpu'].value_counts()
    for gpu, count in unique_unmatched.items():
        print(f"  {gpu} (appears {count}x)")
else:
    print("\n🎉 All GPUs matched successfully!")

# Show sample of matched GPUs with scores
print("\n=== Sample Matched GPUs with Scores ===")
matched_sample = cleaned_df[cleaned_df['GPU_Score'].notna()][['Gpu', 'GPU_Score']].head(15)
print(matched_sample.to_string(index=False))

# Show distribution statistics
print("\n=== GPU Mark Score Statistics ===")
print(cleaned_df['GPU_Score'].describe())

# Show both CPU and GPU scores together
print("\n=== Sample with Both CPU and GPU Scores ===")
both_scores = cleaned_df[['Cpu', 'Prosesor_Score', 'Gpu', 'GPU_Score']].head(10)
print(both_scores.to_string(index=False))

# Save to CSV for inspection
output_file = 'test_output_with_all_scores.csv'
cleaned_df.to_csv(output_file, index=False)
print(f"\n✅ Saved results to: {output_file}")
