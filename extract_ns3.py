from scapy.all import rdpcap, IP
import pandas as pd
import glob

def extract_features(pcap_file, label):
    packets = rdpcap(pcap_file)
    data = []

    prev_time = None

    for pkt in packets:
        if IP in pkt:
            curr_time = float(pkt.time)

            if prev_time is None:
                time_diff = 0
            else:
                time_diff = curr_time - prev_time

            prev_time = curr_time

            packet_rate = 0 if time_diff == 0 else 1 / time_diff

            data.append({
                "packet_size": len(pkt),
                "protocol": pkt[IP].proto,
                "time_diff": time_diff,
                "packet_rate": packet_rate,
                "label": label
            })

    return pd.DataFrame(data)

all_data = []

# Blackhole PCAPs
for file in glob.glob("fyp_ns3_results/*.pcap"):
    if "blackhole" in file:
        df = extract_features(file, 3)
        all_data.append(df)

# Grayhole PCAPs
for file in glob.glob("fyp_ns3_results/*.pcap"):
    if "grayhole" in file:
        df = extract_features(file, 4)
        all_data.append(df)

dataset = pd.concat(all_data, ignore_index=True)

dataset.to_csv("ns3_dataset.csv", index=False)

print("NS3 dataset created:", dataset.shape)
print(dataset["label"].value_counts())
