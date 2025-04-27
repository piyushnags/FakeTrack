import numpy as np
from transformers import pipeline
import re
import matplotlib.pyplot as plt
import torch
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import random

# Load the fine-tuned model on GPU
generator = pipeline("text-generation", model="models/finetuned_model", device=0 if torch.cuda.is_available() else -1)

def initialize_shape_points(initial_location, shape, num_points=5):
    """Generate initial points to hint at the desired shape."""
    lat0, lon0 = initial_location
    points = []
    for i in range(num_points):
        t = (2 * np.pi * i) / num_points
        if shape == "elliptical":
            lat = lat0 + 0.005 * np.cos(t)  # Major axis: 0.005 degrees
            lon = lon0 + 0.0025 * np.sin(t)  # Minor axis: 0.0025 degrees
        elif shape == "circular":
            lat = lat0 + 0.005 * np.cos(t)
            lon = lon0 + 0.005 * np.sin(t)
        elif shape == "straight":
            lat = lat0 + (i * 0.0005)  # Linear increase
            lon = lon0 + (i * 0.0005)
        elif shape == "figure_eight":
            lat = lat0 + 0.005 * np.sin(t)
            lon = lon0 + 0.005 * np.sin(t) * np.cos(t)
        else:
            lat = lat0
            lon = lon0
        points.append((lat, lon))
    return points

def generate_chunk(device_id, activity, start_time_str, initial_points, chunk_size=5):
    """Generate a chunk of points for the sequence."""
    start_time = datetime.strptime(start_time_str, "%I:%M %p")
    prompt = f"Device ID: {device_id}, Activity: {activity}\nSensor data:\n"
    
    # Add the initial points to the prompt for continuity
    for i, (lat, lon) in enumerate(initial_points):
        time = start_time + timedelta(minutes=i)
        time_str = time.strftime("%I:%M %p")
        prompt += f"Time: {time_str}, Location: ({lat:.4f}, {lon:.4f})\n"
    
    # Instruct the model to generate chunk_size points
    prompt += f"Continue the sequence for {chunk_size} time steps, incrementing time by 1 minute each step, including Location and HeartRate for each step.\n"
    max_length = 512  # Safe max_length
    generated = generator(prompt, max_length=max_length, num_return_sequences=1, temperature=0.3)
    return generated[0]["generated_text"]

def extract_locations(text, num_points):
    """Extract locations from the generated text."""
    locations = re.findall(r"Location: \(([0-9.-]+), ([0-9.-]+)\)", text)
    locations = [(float(lat), float(lon)) for lat, lon in locations][:num_points]
    return locations

def generate_full_path(device_id, activity, initial_location, total_points=100, chunk_size=5):
    """Generate a full path with total_points by generating in chunks."""
    # Randomly select a shape to initialize the path
    shape = random.choice(["elliptical", "circular", "straight", "figure_eight"])
    print(f"Initializing path with shape: {shape}")

    # Generate initial points to hint at the shape
    initial_points = initialize_shape_points(initial_location, shape, num_points=chunk_size)
    all_locations = initial_points.copy()
    points_generated = len(initial_points)
    current_location = initial_points[-1]
    start_time_str = "08:00 AM"

    while points_generated < total_points:
        chunk_points = min(chunk_size, total_points - points_generated)
        context_points = all_locations[-chunk_size:] if len(all_locations) >= chunk_size else all_locations
        generated_text = generate_chunk(device_id, activity, start_time_str, context_points, chunk_points)
        chunk_locations = extract_locations(generated_text, chunk_points)
        
        if not chunk_locations:
            break  # Stop if no locations are generated

        all_locations.extend(chunk_locations)
        points_generated += len(chunk_locations)
        if chunk_locations:
            current_location = chunk_locations[-1]  # Update starting point for the next chunk
        
        # Update start time for the next chunk
        start_time = datetime.strptime(start_time_str, "%I:%M %p")
        start_time = start_time + timedelta(minutes=chunk_points)
        start_time_str = start_time.strftime("%I:%M %p")

    # Truncate to exactly total_points
    all_locations = all_locations[:total_points]
    return all_locations

def classify_shape(locations):
    """Classify the shape of the path as elliptical, circular, straight, or figure-eight."""
    if len(locations) < 3:
        return "Unknown"

    lats, lons = zip(*locations)
    lats = np.array(lats)
    lons = np.array(lons)

    # Center the points by subtracting the mean
    lat_center = np.mean(lats)
    lon_center = np.mean(lons)
    lats_centered = lats - lat_center
    lons_centered = lons - lon_center

    # 1. Check for Straight Line
    X = lons_centered.reshape(-1, 1)
    y = lats_centered
    model = LinearRegression().fit(X, y)
    r_squared = model.score(X, y)
    if r_squared > 0.95:  # High R² indicates a straight line
        return "straight"

    # 2. Check for Figure-Eight (lemniscate pattern)
    lat_sign_changes = np.sum(np.diff(np.sign(lats_centered)) != 0)
    lon_sign_changes = np.sum(np.diff(np.sign(lons_centered)) != 0)
    if lat_sign_changes > 2 and lon_sign_changes > 2:  # Multiple crossings
        fft_lats = np.fft.fft(lats_centered)
        fft_lons = np.fft.fft(lons_centered)
        dominant_freq_lats = np.argmax(np.abs(fft_lats[1:len(fft_lats)//2])) + 1
        dominant_freq_lons = np.argmax(np.abs(fft_lons[1:len(fft_lons)//2])) + 1
        if abs(dominant_freq_lats - dominant_freq_lons) <= 1:  # Similar dominant frequencies
            return "figure_eight"

    # 3. Check for Elliptical/Circular
    lat_range = np.max(lats_centered) - np.min(lats_centered)
    lon_range = np.max(lons_centered) - np.min(lons_centered)
    if lat_range > 0 and lon_range > 0:
        aspect_ratio = lat_range / lon_range
        if 0.9 <= aspect_ratio <= 1.1:
            return "circular"
        fft_lats = np.fft.fft(lats_centered)
        fft_lons = np.fft.fft(lons_centered)
        power_lats = np.abs(fft_lats[1:len(fft_lats)//2])
        power_lons = np.abs(fft_lons[1:len(fft_lons)//2])
        if np.max(power_lats) / np.sum(power_lats) > 0.3 and np.max(power_lons) / np.sum(power_lons) > 0.3:
            return "elliptical"

    return "unknown"

def plot_path(locations, device_id, shape):
    """Plot the generated path with the classified shape."""
    lats, lons = zip(*locations)
    plt.figure(figsize=(10, 6))
    plt.plot(lons, lats, label="Generated Path", marker='o', color='blue', markersize=2)
    plt.title(f"Generated Path for Device ID: {device_id} (Shape: {shape})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"benchmark/generated_path_{device_id}.png")
    plt.close()

def evaluate_utility(num_paths=10, num_points=100):
    """Generate multiple full paths and classify their shapes."""
    activities = ["resting", "walking", "running"]
    device_ids = [f"{i:04d}" for i in range(1, 101)]  # Device IDs 0001 to 0100
    expected_shapes = ["elliptical", "circular", "straight", "figure_eight"]
    matches = 0
    results = []

    # Generate 10 random paths
    for i in range(num_paths):
        device_id = random.choice(device_ids)
        activity = random.choice(activities)
        # Random initial location within the range of your dataset
        initial_location = (40.2 + random.uniform(0, 1.0), -74.5 + random.uniform(0, 1.0))

        # Generate a full 100-point path
        print(f"\nGenerating path {i+1}/{num_paths} for Device ID: {device_id}, Activity: {activity}")
        gen_locations = generate_full_path(device_id, activity, initial_location, total_points=num_points, chunk_size=5)
        
        # Debug print to verify the data
        print(f"Number of Generated Points: {len(gen_locations)}")
        print(f"Generated Locations (first 5): {gen_locations[:5]}")

        # Classify the shape of the generated path
        shape = classify_shape(gen_locations)
        print(f"Classified Shape: {shape}")

        # Visualize the path
        plot_path(gen_locations, device_id, shape)

        # Check if the shape matches one of the expected shapes
        if shape in expected_shapes:
            matches += 1
        results.append((device_id, activity, shape))

    # Summarize the results
    print(f"\nUtility Evaluation Summary:")
    print(f"Total Paths Generated: {num_paths}")
    print(f"Paths Matching Expected Shapes: {matches}/{num_paths}")
    print(f"Matching Percentage: {(matches/num_paths)*100:.2f}%")
    print("\nIndividual Results:")
    for device_id, activity, shape in results:
        print(f"Device ID: {device_id}, Activity: {activity}, Shape: {shape}")

    return results

# Example usage
if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    np.random.seed(42)
    num_paths = 5
    evaluate_utility(num_paths=num_paths, num_points=100)