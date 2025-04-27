import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import re
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device set to use {device}")

# Load model and tokenizer
try:
    model = GPT2LMHeadModel.from_pretrained("models/finetuned_model").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Load external database
try:
    external_db = pd.read_csv("data/external_database.csv")
    external_db["device_id"] = external_db["device_id"].astype(str).str.zfill(4)
except FileNotFoundError:
    print("Error: external_database.csv not found in data/")
    exit(1)

# Set up text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Compute perplexity
def compute_perplexity(text):
    try:
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = encodings.input_ids.to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
        return torch.exp(loss).item()
    except Exception as e:
        print(f"Perplexity computation error: {e}")
        return float("inf")

# Generate sequence
def generate_sequence(device_id, activity, initial_location):
    prompt = f"Device ID: {device_id}, Activity: {activity}\nSensor data:\nTime: 10:00 AM, Location: ({initial_location[0]:.4f}, {initial_location[1]:.4f}), HeartRate: 100 bpm\n"
    try:
        generated = generator(prompt, max_length=300, num_return_sequences=1, temperature=0.1)
        return generated[0]["generated_text"]
    except Exception as e:
        print(f"Generation error for Device ID {device_id}: {e}")
        return ""

# Extract locations
def extract_locations(text):
    locations = re.findall(r"Location: \(([0-9.-]+), ([0-9.-]+)\)", text)
    return [(float(lat), float(lon)) for lat, lon in locations]

# Estimate ZIP code with a much coarser grid
def get_zip_code(lat, lon, num_bins=5):
    lat_min = 40.2
    lon_min = -74.5
    bin_width = 0.2  # Larger bin width for lenient estimation
    lat_idx = min(max(int((lat - lat_min) / bin_width), 0), num_bins - 1)
    lon_idx = min(max(int((lon - lon_min) / bin_width), 0), num_bins - 1)
    return f"ZIP_{lat_idx}_{lon_idx}"

def estimate_zip_code(locations):
    if not locations:
        return None
    avg_lat = sum(lat for lat, _ in locations) / len(locations)
    avg_lon = sum(lon for _, lon in locations) / len(locations)
    return get_zip_code(avg_lat, avg_lon)

# Compute distance (km)
def compute_distance(lat1, lon1, lat2, lon2):
    return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5 * 111

# Compute sequence similarity
def compute_sequence_similarity(gen_locations, true_locations):
    if not gen_locations or not true_locations:
        return 0.0
    min_len = min(len(gen_locations), len(true_locations), 5)
    gen_vec = np.array([coord for loc in gen_locations[:min_len] for coord in loc])
    true_vec = np.array([coord for loc in true_locations[:min_len] for coord in loc])
    if len(gen_vec) != len(true_vec):
        return 0.0
    return np.dot(gen_vec, true_vec) / (np.linalg.norm(gen_vec) * np.linalg.norm(true_vec))

# Load training and test data
try:
    with open("data/training.txt", "r") as f:
        train_text = f.read()
    train_device_ids = sorted(list(set(re.findall(r"Device ID: (\d+)", train_text))))
    train_activities = dict(re.findall(r"Device ID: (\d+), Activity: (\w+)", train_text))
except FileNotFoundError:
    print("Error: training.txt not found in data/")
    exit(1)

try:
    with open("data/test.txt", "r") as f:
        test_text = f.read()
    test_device_ids = sorted(list(set(re.findall(r"Device ID: (\d+)", test_text))))
    test_activities = dict(re.findall(r"Device ID: (\d+), Activity: (\w+)", test_text))
except FileNotFoundError:
    print("Error: test.txt not found in data/")
    exit(1)

# Validate device IDs
db_device_ids = set(external_db["device_id"])
train_device_ids = [id for id in train_device_ids if id in db_device_ids]
test_device_ids = [id for id in test_device_ids if id in db_device_ids]

# Evaluate re-identification
def evaluate_reid(device_ids, activities, text_data):
    zip_successes = 0
    dist_successes = 0
    seq_successes = 0
    total_valid = 0
    for device_id in device_ids:
        try:
            user_info = external_db[external_db["device_id"] == device_id]
            if user_info.empty:
                continue
            true_zip = user_info["zip_code"].values[0]
            true_lat = user_info["center_lat"].values[0]
            true_lon = user_info["center_lon"].values[0]
            activity = activities.get(device_id, "running")
            initial_location = (true_lat, true_lon)
            generated_text = generate_sequence(device_id, activity, initial_location)
            locations = extract_locations(generated_text)
            estimated_zip = estimate_zip_code(locations)
            dist = float("inf")
            seq_sim = compute_sequence_similarity(locations, extract_locations(text_data))
            if locations:
                avg_lat = sum(lat for lat, _ in locations) / len(locations)
                avg_lon = sum(lon for _, lon in locations) / len(locations)
                dist = compute_distance(avg_lat, avg_lon, true_lat, true_lon)
                if estimated_zip == true_zip:
                    zip_successes += 1
                if dist < 10:
                    dist_successes += 1
                if seq_sim > 0.9:
                    seq_successes += 1
            total_valid += 1
        except Exception as e:
            print(f"Error for Device ID {device_id}: {e}")
    zip_accuracy = zip_successes / total_valid if total_valid else 0
    dist_accuracy = dist_successes / total_valid if total_valid else 0
    seq_accuracy = seq_successes / total_valid if total_valid else 0
    return zip_accuracy, dist_accuracy, seq_accuracy, total_valid

# Compute accuracies
train_zip_acc, train_dist_acc, train_seq_acc, train_valid = evaluate_reid(train_device_ids, train_activities, train_text)
test_zip_acc, test_dist_acc, test_seq_acc, test_valid = evaluate_reid(test_device_ids, test_activities, test_text)

# Print results
print(f"\nRe-identification ZIP accuracy: Training = {train_zip_acc:.2f}, Test = {test_zip_acc:.2f}")
print(f"Re-identification Distance accuracy: Training = {train_dist_acc:.2f}, Test = {test_dist_acc:.2f}")
print(f"Re-identification Sequence accuracy: Training = {train_seq_acc:.2f}, Test = {test_seq_acc:.2f}")
if train_zip_acc > test_zip_acc or train_dist_acc > test_dist_acc or train_seq_acc > test_seq_acc:
    print("Higher accuracy on training users suggests memorization risk.")
else:
    print("No significant re-identification risk detected.")

# Compute perplexity
train_blocks = [block.strip() for block in train_text.split("\n\n") if block.strip()][:100]
test_blocks = [block.strip() for block in test_text.split("\n\n") if block.strip()][:100]
train_perplexities = [ppl for block in train_blocks if (ppl := compute_perplexity(block)) != float("inf")]
test_perplexities = [ppl for block in test_blocks if (ppl := compute_perplexity(block)) != float("inf")]
avg_train_ppl = sum(train_perplexities) / len(train_perplexities) if train_perplexities else float("inf")
avg_test_ppl = sum(test_perplexities) / len(test_perplexities) if test_perplexities else float("inf")
print(f"\nAverage perplexity: Training = {avg_train_ppl:.2f}, Test = {avg_test_ppl:.2f}")