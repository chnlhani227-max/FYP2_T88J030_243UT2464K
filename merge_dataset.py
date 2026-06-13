import pandas as pd

main_df = pd.read_csv("dataset.csv")
ns3_df = pd.read_csv("ns3_dataset.csv")

# Relabel old attack traffic
# 0 = normal
# 1 = flooding/portscan

merged = pd.concat([main_df, ns3_df], ignore_index=True)

merged.to_csv("final_dataset.csv", index=False)

print("Final dataset created:", merged.shape)
print("\nLabel counts:")
print(merged["label"].value_counts())
