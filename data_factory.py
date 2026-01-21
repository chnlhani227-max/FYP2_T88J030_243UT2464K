import pandas as pd
import numpy as np
import os
os.makedirs("data", exist_ok=True)

def generate(name, n, features, logic):
    data = {k: np.random.normal(v[0], v[1], n) for k, v in features.items()}
    df = pd.DataFrame(data)
    df['Target'] = [logic(df.iloc[i]) for i in range(n)]
    df.to_csv(f"data/{name}.csv", index=False)

generate("Industrial_Motor", 3000, {"Vib_X": (0.5,0.1), "Temp": (65,5)},
         lambda row: 1 if row['Temp'] > 70 else 0)
generate("Smart_Grid", 2500, {"Volt": (230,5), "Freq": (50,0.2)},
         lambda row: 1 if row['Volt'] < 220 or row['Volt'] > 240 else 0)
generate("Network_Traffic", 4000, {"Packet_Size": (500,150), "Rate": (20,5)},
         lambda row: 1 if row['Rate'] > 30 else 0)
generate("Patient_Vitals", 2000, {"BPM": (75,15), "SpO2": (98,1)},
         lambda row: 1 if row['BPM'] > 100 or row['SpO2'] < 95 else 0)
generate("Agri_Sensor", 2000, {"Moisture": (40,10), "PH": (6.5,0.5)},
         lambda row: 1 if row['PH'] < 6.0 or row['PH'] > 7.0 else 0)
generate("Car_Telemetry", 3000, {"RPM": (3000,800), "Engine_Temp": (90,10)},
         lambda row: 1 if row['Engine_Temp'] > 100 else 0)
generate("Drone_Gyro", 3000, {"Gyro_X": (0,2), "Gyro_Y": (0,2), "Alt": (50,10)},
         lambda row: 1 if row['Alt'] < 30 or row['Alt'] > 70 else 0)
generate("Retail_Store", 1500, {"Dwell_Time": (20,10), "Count": (50,20)},
         lambda row: 1 if row['Count'] > 100 else 0)
generate("HVAC_Energy", 2500, {"Power": (1500,200), "Delta_T": (5,2)},
         lambda row: 1 if row['Power'] > 1800 else 0)
generate("Water_Quality", 2000, {"Turbidity": (3,1), "PH": (7,0.5)},
         lambda row: 1 if row['Turbidity'] > 5 or row['PH'] < 6.5 or row['PH'] > 7.5 else 0)
