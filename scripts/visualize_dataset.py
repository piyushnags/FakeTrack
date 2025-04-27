import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from datetime import datetime
import matplotlib.colors as mcolors
from matplotlib import colormaps
import matplotlib.cm as cm
import re

# Ensure benchmark directories exist
os.makedirs("benchmark", exist_ok=True)
os.makedirs("benchmark/paths", exist_ok=True)

# Function to parse text files (training.txt, test.txt, validation.txt)
def parse_text_file(filename):
    data = []
    with open(filename, "r") as f:
        text = f.read()
        blocks = text.strip().split("\n\n")
        for block in blocks:
            lines = block.split("\n")
            device_id = re.search(r"Device ID: (\d+)", lines[0]).group(1)
            activity = re.search(r"Activity: (\w+)", lines[0]).group(1)
            for line in lines[2:]:
                match = re.search(r"Time: (.*?),\s*Location: \(([0-9.-]+),\s*([0-9.-]+)\),\s*HeartRate: (\d+)", line)
                if match:
                    time_str, lat, lon, hr = match.groups()
                    try:
                        time = datetime.strptime(time_str, "%I:%M %p")
                    except ValueError:
                        continue
                    data.append({
                        "device_id": device_id,
                        "activity": activity,
                        "time": time,
                        "latitude": float(lat),
                        "longitude": float(lon),
                        "heart_rate": int(hr)
                    })
    return pd.DataFrame(data)

# Load and combine datasets
try:
    train_df = parse_text_file("data/training.txt")
    val_df = parse_text_file("data/validation.txt")
    test_df = parse_text_file("data/test.txt")
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

# Load external database
try:
    external_db = pd.read_csv("data/external_database.csv")
    external_db["device_id"] = external_db["device_id"].astype(str).str.zfill(4)
except FileNotFoundError:
    print("Error: external_database.csv not found in data/")
    exit(1)

# Summary statistics
stats = {
    "Total Users": len(df["device_id"].unique()),
    "Training Users": len(train_df["device_id"].unique()),
    "Validation Users": len(val_df["device_id"].unique()),
    "Test Users": len(test_df["device_id"].unique()),
    "Data Points per User": len(df) // len(df["device_id"].unique()),
    "Latitude Range": f"{df['latitude'].min():.4f} to {df['latitude'].max():.4f}",
    "Longitude Range": f"{df['longitude'].min():.4f} to {df['longitude'].max():.4f}",
    "Heart Rate Mean": df["heart_rate"].mean(),
    "Heart Rate Range": f"{df['heart_rate'].min()} to {df['heart_rate'].max()}"
}
activity_counts = df.groupby("activity")["device_id"].nunique().to_dict()

# Save summary statistics
stats_df = pd.DataFrame({
    "Metric": list(stats.keys()) + ["Activity: " + k for k in activity_counts.keys()],
    "Value": list(stats.values()) + list(activity_counts.values())
})
stats_df.to_csv("benchmark/summary_statistics.csv", index=False)

# Filter for jogging/running paths (exclude resting)
active_df = df[df["activity"].isin(["walking", "running"])]

# Select 5 random users for visualization
sample_users = np.random.choice(active_df["device_id"].unique(), size=25, replace=False)

# Set up colormap for heart rate
norm = mcolors.Normalize(vmin=df["heart_rate"].min(), vmax=df["heart_rate"].max())
cmap = colormaps["viridis"]  # Updated to use modern colormap access
scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)

# Visualize per-user jogging/running paths with heart rate coloring
for user_id in sample_users:
    user_df = active_df[active_df["device_id"] == user_id]
    # Sort by time and take first 30 minutes (~30 points)
    user_df = user_df.sort_values("time").head(30)
    if len(user_df) < 2:
        continue
    
    plt.figure(figsize=(8, 6))
    ax = plt.gca()  # Get current axes
    
    # Plot path with color varying by heart rate
    for i in range(len(user_df) - 1):
        plt.plot(
            user_df["longitude"].iloc[i:i+2],
            user_df["latitude"].iloc[i:i+2],
            color=scalar_map.to_rgba(user_df["heart_rate"].iloc[i]),
            linewidth=2
        )
    # Mark start and end points
    plt.scatter(user_df["longitude"].iloc[0], user_df["latitude"].iloc[0], c="green", label="Start", s=100)
    plt.scatter(user_df["longitude"].iloc[-1], user_df["latitude"].iloc[-1], c="red", label="End", s=100)
    
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"User {user_id} Path ({user_df['activity'].iloc[0].capitalize()})")
    plt.legend()
    plt.grid(True)
    # Add colorbar, explicitly linked to the axes
    plt.colorbar(scalar_map, ax=ax, label="Heart Rate (bpm)")
    plt.savefig(f"benchmark/paths/user_{user_id}.png")
    plt.close()

# Heart rate heatmap
plt.figure(figsize=(10, 8))
heatmap_data, xedges, yedges = np.histogram2d(
    df["longitude"], df["latitude"], bins=50, weights=df["heart_rate"], density=True
)
sns.heatmap(heatmap_data.T, cmap="YlOrRd", cbar_kws={"label": "Heart Rate Intensity"})
plt.xlabel("Longitude Bin")
plt.ylabel("Latitude Bin")
plt.title("Heart Rate Density Heatmap")
plt.savefig("benchmark/heart_rate_heatmap.png")
plt.close()

print("Visualizations and statistics saved in 'benchmark/' directory.")