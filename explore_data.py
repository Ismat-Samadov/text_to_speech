import pandas as pd
import os

# Get first parquet file
data_dir = 'data'
files = sorted(os.listdir(data_dir))
first_file = os.path.join(data_dir, files[0])

print(f"Reading: {first_file}\n")

# Read parquet file
df = pd.read_parquet(first_file)

print("="*80)
print("DATASET SCHEMA")
print("="*80)
print(f"\nShape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nColumn Data Types:")
print(df.dtypes)

print("\n" + "="*80)
print("SAMPLE DATA (first 5 rows)")
print("="*80)
print(df.head())

print("\n" + "="*80)
print("TEXT STATISTICS")
print("="*80)
if 'text' in df.columns:
    print(f"Total samples: {len(df)}")
    print(f"Unique texts: {df['text'].nunique()}")
    print(f"\nText length statistics:")
    text_lengths = df['text'].str.len()
    print(f"  Min: {text_lengths.min()}")
    print(f"  Max: {text_lengths.max()}")
    print(f"  Mean: {text_lengths.mean():.2f}")
    print(f"  Median: {text_lengths.median():.2f}")

print("\n" + "="*80)
print("SAMPLE TRANSCRIPTIONS")
print("="*80)
if 'text' in df.columns:
    for i, text in enumerate(df['text'].head(10), 1):
        print(f"{i}. {text}")

print("\n" + "="*80)
print("AUDIO INFORMATION")
print("="*80)
if 'audio' in df.columns:
    print(f"Audio column type: {type(df['audio'].iloc[0])}")
    print(f"Audio data structure:")
    print(df['audio'].iloc[0])
