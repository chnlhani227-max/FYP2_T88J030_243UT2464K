import pandas as pd
import numpy as np
import os
import glob

def merge_cic_robust():
    # This looks into EVERY subfolder inside Downloads for .csv files
    search_path = os.path.expanduser("~/Downloads/**/*.csv")
    all_files = glob.glob(search_path, recursive=True)

    # Filter out files that are NOT CICIDS (we only want the ones with 'WorkingHours')
    cic_files = [f for f in all_files if "WorkingHours" in f]

    if not cic_files:
        print("❌ Error: No CICIDS files found! Did you extract the zip?")
        print("Looking for files containing 'WorkingHours' in ~/Downloads")
        return

    li = []
    print(f"⏳ Found {len(cic_files)} files. Starting merge...")

    for filename in cic_files:
        print(f"   Reading: {os.path.basename(filename)}")
        try:
            # Using low_memory and specific encoding for CICIDS
            df = pd.read_csv(filename, index_col=None, header=0, encoding='cp1252', low_memory=False)
            li.append(df)
        except Exception as e:
            print(f"   ⚠️ Could not read {filename}: {e}")

    print("🔗 Merging all days...")
    master_df = pd.concat(li, axis=0, ignore_index=True)

    print("🧹 Cleaning data...")
    master_df.columns = master_df.columns.str.strip()
    master_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    master_df.dropna(inplace=True)

    # Ensure Label column exists
    if 'Label' not in master_df.columns:
        print("❌ Error: 'Label' column not found in these files!")
        return

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    master_df['Label'] = le.fit_transform(master_df['Label'].astype(str))

    output_path = "data/CICIDS_Big_Data_Master.csv"
    # Take a 100,000 row sample (Safe size for your RAM)
    df_sample = master_df.sample(n=min(100000, len(master_df)), random_state=42)
    df_sample.to_csv(output_path, index=False)
    
    print(f"✅ SUCCESS! Created {output_path} with {len(df_sample)} rows.")
    print(f"   Attacks detected: {list(le.classes_)}")

if __name__ == "__main__":
    merge_cic_robust()
