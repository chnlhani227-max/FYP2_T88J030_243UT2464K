from scapy.all import rdpcap, IP
import pandas as pd

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

            # Calculate packet rate
            packet_rate = 0 if time_diff == 0 else 1 / time_diff

            data.append({
                "packet_size": len(pkt),
                "protocol": pkt[IP].proto,
                "time_diff": time_diff,
                "packet_rate": packet_rate,
                "label": label
            })

    return pd.DataFrame(data)

# Normal traffic
normal = extract_features("normal.pcap", 0)

# Flooding / Port Scan traffic
attack = extract_features("attack.pcap", 1)

# Merge datasets
dataset = pd.concat([normal, attack], ignore_index=True)

# Save dataset
dataset.to_csv("dataset.csv", index=False)

print("Normal packets:", len(normal))
print("Attack packets:", len(attack))
print("Dataset created:", dataset.shape)
print("Saved as dataset.csv")
