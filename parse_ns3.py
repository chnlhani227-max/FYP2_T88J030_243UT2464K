import pandas as pd
import os

# Path to the ns3 trace file
trace_path = os.path.expanduser("~/ns-allinone-3.40/ns-3.40/wifi7_trace.tr")

def parse_trace():
    print("⏳ Parsing ns-3 Trace file for WiFi 7 features...")
    data = []
    
    if not os.path.exists(trace_path):
        print("❌ Error: wifi7_trace.tr not found! Run the ns-3 script first.")
        return

    with open(trace_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) > 10 and parts[0] in ['r', 'd', 't']: # received, dropped, transmitted
                # Extracting simulated WiFi 7 features for Hypothesis 1
                try:
                    time = float(parts[1])
                    size = int(parts[parts.index('PayloadSize:')+1]) if 'PayloadSize:' in parts else 0
                    # Simulate WiFi 7 specific features (MLO latency and 6GHz noise)
                    # In a real project, you'd extract these from specific trace headers
                    mlo_latency = time * 0.01 
                    channel_width = 320 # WiFi 7 feature
                    freq = 6.0 # 6GHz Band
                    
                    label = 1 if time > 1.0 else 0 # Attack started at 1.0s
                    
                    data.append([time, size, mlo_latency, channel_width, freq, label])
                except:
                    continue

    df = pd.DataFrame(data, columns=['Time', 'PacketSize', 'MLO_Latency', 'Bandwidth_MHz', 'Frequency_GHz', 'Label'])
    df.to_csv("data/NS3_WiFi7_Simulated.csv", index=False)
    print("✅ SUCCESS! Created data/NS3_WiFi7_Simulated.csv")

if __name__ == "__main__":
    parse_trace()
