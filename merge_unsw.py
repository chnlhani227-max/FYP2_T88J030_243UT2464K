import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# Official Headers for UNSW-NB15 Raw Records
headers = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 
    'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 
    'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 
    'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 
    'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'Label'
]

def merge_unsw_robust():
    folder = os.path.expanduser("~/Downloads")
    files = [f"{folder}/UNSW-NB15_{i}.csv" for i in range(1, 5)]
    
    combined_list = []
    for file in files:
        if os.path.exists(file):
            print(f"⏳ Reading {file}...")
            # Use low_memory=False to handle mixed types
            df = pd.read_csv(file, names=headers, low_memory=False)
            combined_list.append(df)
    
    if not combined_list:
        print("❌ Error: No files found in Downloads! Check filenames.")
        return

    print("🔗 Merging...")
    master_df = pd.concat(combined_list, ignore_index=True)

    # 1. DROP useless columns for AI
    print("🧹 Dropping non-numeric columns...")
    drop_cols = ['srcip', 'sport', 'dstip', 'dsport', 'Stime', 'Ltime', 'attack_cat']
    master_df.drop(columns=drop_cols, errors='ignore', inplace=True)

    # 2. FIX MIXED TYPES (This is usually why it fails)
    # Convert anything that looks like a number but is a string
    for col in master_df.columns:
        if col not in ['proto', 'state', 'service']: # These are definitely text
            master_df[col] = pd.to_numeric(master_df[col], errors='coerce')

    # 3. ENCODE TEXT COLUMNS
    print("🏷️ Encoding text columns (proto, state, service)...")
    le = LabelEncoder()
    for col in ['proto', 'state', 'service']:
        master_df[col] = le.fit_transform(master_df[col].astype(str))

    # 4. FINAL CLEANUP
    print("🧼 Removing NaNs and Infinity...")
    master_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    master_df.dropna(inplace=True)

    # 5. SAVE
    output_path = "data/UNSW_Big_Data_Master.csv"
    # Take 200,000 rows (Balanced for performance and accuracy)
    df_sample = master_df.sample(n=min(200000, len(master_df)), random_state=42)
    df_sample.to_csv(output_path, index=False)
    
    print(f"✅ SUCCESS! Created {output_path}")
    print(f"   Final shape: {df_sample.shape}")

if __name__ == "__main__":
    merge_unsw_robust()
