import pandas as pd
from faker import Faker
import random
import math
from datetime import datetime, timedelta

# Initialize Faker for synthetic data
faker = Faker()

# Parameters
num_users = 100
points_per_user = 100

# Generate user metadata with sensitive information
users = []
for i in range(num_users):
    device_id = f"{i:04d}"  # Unique device ID (e.g., "0001")
    name = faker.name()
    age = random.randint(18, 65)
    gender = random.choice(["Male", "Female", "Other"])
    zip_code = f"{random.randint(10000, 99999)}"
    activity = random.choice(["walking", "running", "resting"])
    users.append({
        "device_id": device_id,
        "name": name,
        "age": age,
        "gender": gender,
        "zip": zip_code,
        "activity": activity
    })

# Split users into train, validation, and test sets
random.shuffle(users)
train_users = users[:80]
val_users = users[80:90]
test_users = users[90:]

# Save external database (sensitive info) to CSV
external_db = pd.DataFrame(users)[["device_id", "name", "age", "gender", "zip"]]
external_db.to_csv("data/external_database.csv", index=False)

# Function to generate anonymized sensor data for a user
def generate_user_data(user):
    activity = user["activity"]
    # Activity-specific parameters
    if activity == "walking":
        a, b = 0.005, 0.0025
        hr_min, hr_max = 100, 120
    elif activity == "running":
        a, b = 0.02, 0.01
        hr_min, hr_max = 140, 180
    else:  # resting
        a, b = 0.0001, 0.0001
        hr_min, hr_max = 60, 80
    
    # Random center location near NYC
    center_lat = 40.7128 + random.uniform(-0.1, 0.1)
    center_lon = -74.0060 + random.uniform(-0.1, 0.1)
    
    # Start time
    start_time = datetime(2023, 1, 1, 10, 0, 0)
    
    # Generate sensor data points
    data_points = []
    for t in range(points_per_user):
        time = start_time + timedelta(minutes=t)
        time_str = time.strftime("%H:%M %p")
        angle = 2 * math.pi * (t / points_per_user)
        lat = center_lat + a * math.cos(angle)
        lon = center_lon + b * math.sin(angle)
        hr = random.randint(hr_min, hr_max)
        data_points.append(
            f"Time: {time_str}, Location: ({lat:.4f}, {lon:.4f}), HeartRate: {hr} bpm"
        )
    
    # Construct anonymized user block
    user_block = (
        f"Device ID: {user['device_id']}, Activity: {activity}\n"
        f"Sensor data:\n" + "\n".join(data_points)
    )
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