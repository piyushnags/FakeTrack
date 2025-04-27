import pandas as pd
from faker import Faker
import random
import math
from datetime import datetime, timedelta

# Initialize Faker for generating fake names
faker = Faker()

# Parameters
num_users = 10
points_per_user = 100
total_points = num_users * points_per_user

# Generate user names (PII)
users = [faker.name() for _ in range(num_users)]

# Generate center locations for each user (around New York City coordinates)
centers = [(40.7128 + random.uniform(-0.1, 0.1), -74.0060 + random.uniform(-0.1, 0.1)) for _ in range(num_users)]

# Generate synthetic data points
data = []
start_time = datetime(2023, 1, 1, 10, 0, 0)
for user_idx, user in enumerate(users):
    center_lat, center_lon = centers[user_idx]
    a = 0.01  # Semi-major axis (~1 km)
    b = 0.005  # Semi-minor axis (~0.5 km)
    for t in range(points_per_user):
        time = start_time + timedelta(minutes=t)
        time_str = time.strftime("%H:%M %p")
        angle = 2 * math.pi * (t / points_per_user)
        lat = center_lat + a * math.cos(angle)
        lon = center_lon + b * math.sin(angle)
        hr = random.randint(120, 160)  # Heart rate typical for jogging
        text = f"User: {user}, Time: {time_str}, Location: ({lat:.4f}, {lon:.4f}), HeartRate: {hr} bpm"
        data.append(text)

# Split data into training and test sets
random.shuffle(data)
train_data = data[:total_points // 2]  # 500 points
test_data = data[total_points // 2:]   # 500 points

# Save to files in the data directory
with open("data/training.txt", "w") as f:
    for line in train_data:
        f.write(line + "\n")

with open("data/test.txt", "w") as f:
    for line in test_data:
        f.write(line + "\n")

print("Synthetic dataset generated and saved to data/training.txt and data/test.txt")
