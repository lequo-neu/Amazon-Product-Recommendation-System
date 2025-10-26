import polars as pl
import numpy as np
from pathlib import Path

# Load test data
test_path = Path('/Users/kevin/Documents/GitHub/Python/VESKL/Personal/NEU/NEU/NEU_7275/Prj/Amazon-Product-Recommendation-System/data/processed/Electronics.5core.train.parquet')
test = pl.read_parquet(test_path)

print("="*70)
print("ANALYZING RATING DISTRIBUTION FOR THRESHOLD RECOMMENDATION")
print("="*70)

print(f"\nTest set stats:")
print(f"Total ratings: {test.height:,}")
print(f"Unique users:  {test['user_id'].n_unique():,}")
print(f"Unique items:  {test['parent_asin'].n_unique():,}")

# USER RATING DISTRIBUTION
print("\n" + "="*70)
print("USER RATING DISTRIBUTION")
print("="*70)

user_counts = test.group_by('user_id').agg(pl.len().alias('count'))

print(f"\nUser rating statistics:")
print(f"  Min:    {user_counts['count'].min():.0f}")
print(f"  25%:    {user_counts['count'].quantile(0.25):.0f}")
print(f"  Median: {user_counts['count'].median():.0f}")
print(f"  75%:    {user_counts['count'].quantile(0.75):.0f}")
print(f"  Max:    {user_counts['count'].max():.0f}")
print(f"  Mean:   {user_counts['count'].mean():.2f}")

# User distribution by rating count
user_dist = user_counts.group_by('count').agg(pl.len().alias('num_users')).sort('count')
print(f"\nUser distribution (first 15 buckets):")
for row in user_dist.head(15).iter_rows(named=True):
    pct = row['num_users'] / len(user_counts) * 100
    print(f"  {row['count']:3d} ratings: {row['num_users']:6,} users ({pct:5.2f}%)")

# ITEM RATING DISTRIBUTION
print("\n" + "="*70)
print("ITEM RATING DISTRIBUTION")
print("="*70)

item_counts = test.group_by('parent_asin').agg(pl.len().alias('count'))

print(f"\nItem rating statistics:")
print(f"  Min:    {item_counts['count'].min():.0f}")
print(f"  25%:    {item_counts['count'].quantile(0.25):.0f}")
print(f"  Median: {item_counts['count'].median():.0f}")
print(f"  75%:    {item_counts['count'].quantile(0.75):.0f}")
print(f"  Max:    {item_counts['count'].max():.0f}")
print(f"  Mean:   {item_counts['count'].mean():.2f}")

# Item distribution by rating count
item_dist = item_counts.group_by('count').agg(pl.len().alias('num_items')).sort('count')
print(f"\nItem distribution (first 15 buckets):")
for row in item_dist.head(15).iter_rows(named=True):
    pct = row['num_items'] / len(item_counts) * 100
    print(f"  {row['count']:3d} ratings: {row['num_items']:6,} items ({pct:5.2f}%)")

# RECOMMENDED THRESHOLDS
print("\n" + "="*70)
print("RECOMMENDED THRESHOLDS (Based on 5-core dataset)")
print("="*70)

# User thresholds
user_counts_sorted = sorted(user_counts['count'].to_list())
user_p20 = user_counts_sorted[int(len(user_counts_sorted) * 0.2)]
user_p50 = user_counts_sorted[int(len(user_counts_sorted) * 0.5)]
user_p80 = user_counts_sorted[int(len(user_counts_sorted) * 0.8)]

print(f"\nUSER SCENARIOS:")
print(f"  New User:    0 ratings (not in test)")
print(f"  Cold User:   1-{user_p20} ratings   (bottom 20%)")
print(f"  Warm User:   {user_p20+1}-{user_p80} ratings  (20-80%)")
print(f"  Active User: >{user_p80} ratings    (top 20%)")

# Item thresholds
item_counts_sorted = sorted(item_counts['count'].to_list())
item_p20 = item_counts_sorted[int(len(item_counts_sorted) * 0.2)]
item_p50 = item_counts_sorted[int(len(item_counts_sorted) * 0.5)]
item_p80 = item_counts_sorted[int(len(item_counts_sorted) * 0.8)]

print(f"\nITEM SCENARIOS:")
print(f"  New Item:     0 ratings (not in test)")
print(f"  Cold Item:    1-{item_p20} ratings   (bottom 20%)")
print(f"  Warm Item:    {item_p20+1}-{item_p80} ratings  (20-80%)")
print(f"  Popular Item: >{item_p80} ratings    (top 20%)")

# Count items in each range for cold items
cold_item_count = len(item_counts.filter(pl.col('count') <= item_p20))
print(f"\nEstimated cold items (1-{item_p20} ratings): {cold_item_count:,} items")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\n✓ Recommended Cold User threshold:  1-{user_p20} ratings")
print(f"✓ Recommended Cold Item threshold:  1-{item_p20} ratings")
print(f"\n💡 Use these thresholds in app.py get_cold_items() function")
print("="*70)
