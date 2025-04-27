import random
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# ZIP code assignment with coarse 5x5 grid (unchanged from original commit)
def get_zip_code(lat, lon, num_bins=5):
    lat_min = 40.2
    lon_min = -74.5
    bin_width = 0.2
    lat_idx = min(max(int((lat - lat_min) / bin_width), 0), num_bins - 1)
    lon_idx = min(max(int((lon - lon_min) / bin_width), 0), num_bins - 1)
    return f"ZIP_{lat_idx}_{lon_idx}"

# Path generation functions for 4 distinct shapes (removed random_walk)
def generate_elliptical_path(center_lat, center_lon, a, b, num_points):
    angles = np.sort(np.random.uniform(0, 2 * np.pi, num_points))  # Sort angles for smoother plotting
    lat = center_lat + a * np.cos(angles)
    lon = center_lon + b * np.sin(angles)
    return lat, lon

def generate_circular_path(center_lat, center_lon, radius, num_points):
    angles = np.sort(np.random.uniform(0, 2 * np.pi, num_points))
    lat = center_lat + radius * np.cos(angles)
    lon = center_lon + radius * np.sin(angles)
    return lat, lon

def generate_straight_path(start_lat, start_lon, end_lat, end_lon, num_points):
    lat = np.linspace(start_lat, end_lat, num_points)
    lon = np.linspace(start_lon, end_lon, num_points)
    return lat, lon

def generate_figure_eight_path(center_lat, center_lon, scale, num_points):
    theta = np.linspace(0, 2 * np.pi, num_points)
    lat = center_lat + scale * np.sin(theta)
    lon = center_lon + scale * np.sin(theta) * np.cos(theta)
    return lat, lon

# Generate user data with varied path shapes (no random_walk)
def generate_user_data(user_id, num_points=100):
    device_id = f"{user_id:04d}"
    center_lat = 40.7128 + random.uniform(-0.5, 0.5)
    center_lon = -74.0060 + random.uniform(-0.5, 0.5)
    zip_code = get_zip_code(center_lat, center_lon)
    name = f"User_{device_id}"
    age = random.randint(18, 80)
    gender = random.choice(["Male", "Female", "Other"])
    activity = random.choice(["resting", "walking", "running"])

    # Assign path shapes: only elliptical, circular, straight, figure_eight
    path_shape = random.choice(["elliptical", "circular", "straight", "figure_eight"])
    
    # Debug print to confirm path shape assignment
    print(f"User {device_id} (ID: {user_id}): Path Shape Assigned = {path_shape}")

    # Adjust path parameters based on activity
    if activity == "resting":
        scale = 0.0001  # Minimal movement
        hr_mean, hr_std = 70, 5
    elif activity == "walking":
        scale = 0.005
        hr_mean, hr_std = 100, 10
    else:  # running
        scale = 0.02
        hr_mean, hr_std = 130, 15

    # Generate path based on shape and scale
    if path_shape == "elliptical":
        a, b = scale, scale / 2
        lat, lon = generate_elliptical_path(center_lat, center_lon, a, b, num_points)
        print(f"User {device_id}: Generating elliptical path with a={a}, b={b}")
    elif path_shape == "circular":
        radius = scale
        lat, lon = generate_circular_path(center_lat, center_lon, radius, num_points)
        print(f"User {device_id}: Generating circular path with radius={radius}")
    elif path_shape == "straight":
        end_lat = center_lat + scale * random.uniform(-1, 1)
        end_lon = center_lon + scale * random.uniform(-1, 1)
        lat, lon = generate_straight_path(center_lat, center_lon, end_lat, end_lon, num_points)
        print(f"User {device_id}: Generating straight path from ({center_lat}, {center_lon}) to ({end_lat}, {end_lon})")
    else:  # figure_eight
        lat, lon = generate_figure_eight_path(center_lat, center_lon, scale, num_points)
        print(f"User {device_id}: Generating figure-eight path with scale={scale}")

    # Debug print: Show the first 5 coordinates to verify the path shape
    print(f"User {device_id}: First 5 coordinates:")
    for i in range(min(5, num_points)):
        print(f"  ({lat[i]:.4f}, {lon[i]:.4f})")

    data = []
    start_time = datetime(2025, 4, 27, 8, 0)  # Arbitrary start
    for i in range(num_points):
        t = start_time + timedelta(minutes=i)
        heart_rate = int(np.random.normal(hr_mean, hr_std))
        heart_rate = max(40, min(200, heart_rate))  # Bound heart rate
        data.append({
            "device_id": device_id,
            "time": t.strftime("%I:%M %p"),
            "latitude": lat[i],
            "longitude": lon[i],
            "heart_rate": heart_rate,
            "activity": activity,
            "zip_code": zip_code,
            "name": name,
            "age": age,
            "gender": gender,
            "path_shape": path_shape  # Track shape for analysis
        })
    return data

# Generate dataset
num_users = 100
all_data = []
for user_id in range(1, num_users + 1):
    user_data = generate_user_data(user_id)
    all_data.extend(user_data)

# Convert to DataFrame
df = pd.DataFrame(all_data)

# Split into training, validation, test
train_df = df[df["device_id"].isin([f"{i:04d}" for i in range(1, 81)])]
val_df = df[df["device_id"].isin([f"{i:04d}" for i in range(81, 91)])]
test_df = df[df["device_id"].isin([f"{i:04d}" for i in range(91, 101)])]

# Save datasets (text files and external database)
def save_text_data(df, filename):
    with open(filename, "w") as f:
        for device_id in sorted(df["device_id"].unique()):
            user_df = df[df["device_id"] == device_id]
            activity = user_df["activity"].iloc[0]
            f.write(f"Device ID: {device_id}, Activity: {activity}\nSensor data:\n")
            for _, row in user_df.iterrows():
                f.write(f"Time: {row['time']}, Location: ({row['latitude']:.4f}, {row['longitude']:.4f}), HeartRate: {row['heart_rate']} bpm\n")
            f.write("\n")

os.makedirs("data", exist_ok=True)
save_text_data(train_df, "data/training.txt")
save_text_data(val_df, "data/validation.txt")
save_text_data(test_df, "data/test.txt")

external_db = df[["device_id", "name", "age", "gender", "zip_code", "path_shape"]].drop_duplicates()
external_db["center_lat"] = df.groupby("device_id")["latitude"].mean().values
external_db["center_lon"] = df.groupby("device_id")["longitude"].mean().values
external_db.to_csv("data/external_database.csv", index=False)

print("Dataset generated with 4 distinct path shapes: elliptical, circular, straight, figure_eight.")