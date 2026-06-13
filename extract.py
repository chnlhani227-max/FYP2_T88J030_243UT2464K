from scapy.all import rdpcap, IP
import pandas as pd

def extract_features(pcap_file, label):
    packets = rdpcap(pcap_file)
    data = []
    prev_time = None

    for pkt in packets:
        if IP in pkt:
            curr_time = float(pkt.time)
            time_diff = 0 if prev_time is None else curr_time - prev_time
            prev_time = curr_time

            data.append({
                "packet_size": len(pkt),
                "protocol": pkt[IP].proto,
                "time_diff": time_diff,
                "label": label
            })

    return pd.DataFrame(data)

normal = extract_features("normal.pcap", 0)
attack = extract_features("attack.pcap", 1)

dataset = pd.concat([normal, attack], ignore_index=True)
dataset.to_csv("dataset.csv", index=False)

print("Normal packets:", len(normal))
print("Attack packets:", len(attack))
print("Dataset created:", dataset.shape)
print("Saved as dataset.csv")
