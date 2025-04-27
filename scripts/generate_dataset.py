import pandas as pd
from faker import Faker
import random
import math
from datetime import datetime, timedelta

# Initialize Faker for generating fake names
faker = Faker()

# Parameters
num_users = 100
points_per_user = 100
activities = ["walking", "running", "resting"]
hr_ranges = {
    "walking": (100, 120),
    "running": (140, 180),
    "resting": (60, 80)
}

# Function to compute ZIP code based on latitude and longitude
def get_zip_code(lat, lon, num_bins=10):
    lat_min = 40.6
    lon_min = -74.1
    bin_width = 0.02
    lat_idx = min(max(int((lat - lat_min) / bin_width), 0), num_bins - 1)
    lon_idx = min(max(int((lon - lon_min) / bin_width), 0), num_bins - 1)
    return f"ZIP_{lat_idx}_{lon_idx}"

# Generate user metadata with sensitive information
users = []
for i in range(num_users):
    device_id = f"{i:04d}"  # Unique device ID (e.g., "0001")
    name = faker.name()
    age = random.randint(18, 65)
    gender = random.choice(["Male", "Female", "Other"])
    activity = random.choice(activities)
    center_lat = 40.7128 + random.uniform(-0.1, 0.1)
    center_lon = -74.0060 + random.uniform(-0.1, 0.1)
    zip_code = get_zip_code(center_lat, center_lon)
    users.append({
        "device_id": device_id,
        "name": name,
        "age": age,
        "gender": gender,
        "activity": activity,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "zip_code": zip_code
    })

# Split users into train, validation, and test sets
random.shuffle(users)
train_users = users[:80]
val_users = users[80:90]
test_users = users[90:]

# Save external database (sensitive info) to CSV
external_db = pd.DataFrame(users)[["device_id", "name", "age", "gender", "zip_code"]]
external_db.to_csv("data/external_database.csv", index=False)

# Function to generate anonymized sensor data for a user
def generate_user_data(user):
    activity = user["activity"]
    hr_min, hr_max = hr_ranges[activity]
    center_lat = user["center_lat"]
    center_lon = user["center_lon"]
    # Define activity-specific path sizes
    if activity == "walking":
        a, b = 0.005, 0.0025  # Smaller path
    elif activity == "running":
        a, b = 0.02, 0.01     # Larger path
    else:  # resting
        a, b = 0.0001, 0.0001 # Minimal movement
    # Random start time for diversity
    start_time = datetime(2023, 1, 1, 10, 0, 0) + timedelta(minutes=random.randint(0, 1440))
    data_points = []
    for t in range(points_per_user):
        time = start_time + timedelta(minutes=t)
        time_str = time.strftime("%H:%M %p")
        angle = 2 * math.pi * (t / points_per_user)
        lat = center_lat + a * math.cos(angle)
        lon = center_lon + b * math.sin(angle)
        hr = random.randint(hr_min, hr_max)
        data_points.append(f"Time: {time_str}, Location: ({lat:.4f}, {lon:.4f}), HeartRate: {hr} bpm")
    user_block = f"Device ID: {user['device_id']}, Activity: {activity}\nSensor data:\n" + "\n".join(data_points)
    return user_block

# Write anonymized data to files
with open("data/training.txt", "w") as f:
    for user in train_users:
        f.write(generate_user_data(user) + "\n\n")
with open("data/validation.txt", "w") as f:
    for user in val_users:
        f.write(generate_user_data(user) + "\n\n")
with open("data/test.txt", "w") as f:
    for user in test_users:
        f.write(generate_user_data(user) + "\n\n")

print("Datasets generated: training.txt, validation.txt, test.txt, external_database.csv")