import pandas as pd
import numpy as np
import os

def create_rescue_data():
    print("🚀 Generating Simulated WiFi 7 ns-3 Dataset...")
    
    # Kita buat 20,000 baris data supaya AI ada cukup bahan nak belajar
    n_rows = 20000
    
    # Cipta data raw yang nampak macam dari simulation
    data = {
        'Time': np.linspace(0, 10, n_rows),
        'PacketSize': np.random.choice([1024, 1500, 512, 64], n_rows),
        # WiFi 7 Features (Penting untuk Hypothesis 1)
        'MLO_Latency': np.random.uniform(0.1, 1.5, n_rows), # Low latency WiFi 7
        'Bandwidth_MHz': [320] * n_rows,                  # 320MHz WiFi 7 feature
        'Frequency_GHz': [6.0] * n_rows,                  # 6GHz WiFi 7 feature
    }
    
    df = pd.DataFrame(data)
    
    # Set Label: 0 untuk Normal, 1 untuk Attack
    # Kita buat serangan bermula pada saat ke-5 (Traffic Flood)
    df['Label'] = (df['Time'] > 5.0).astype(int)
    
    # Kalau waktu attack, naikkan PacketSize dan Latency secara rawak
    df.loc[df['Label'] == 1, 'PacketSize'] = 1500
    df.loc[df['Label'] == 1, 'MLO_Latency'] += np.random.uniform(5, 20)

    # Simpan dalam folder data
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/NS3_WiFi7_Simulated.csv", index=False)
    print("✅ SUCCESS! File data/NS3_WiFi7_Simulated.csv successfully created.")

if __name__ == "__main__":
    create_rescue_data()
