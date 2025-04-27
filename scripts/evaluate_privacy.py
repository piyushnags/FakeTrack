import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import re
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device set to use {device}")

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained("models/finetuned_model").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()

# Load external database
try:
    external_db = pd.read_csv("data/external_database.csv")
    external_db["device_id"] = external_db["device_id"].astype(str).str.zfill(4)
except FileNotFoundError:
    print("Error: external_database.csv not found in data/")
    exit(1)

# Set up text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Function to compute perplexity
def compute_perplexity(text):
    try:
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = encodings.input_ids.to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
        return torch.exp(loss).item()
    except Exception as e:
        print(f"Perplexity computation error: {e}")
        return float("inf")

# Function to generate a sequence for a device ID
def generate_sequence(device_id, activity, initial_location=None):
    prompt = f"Device ID: {device_id}, Activity: {activity}\nSensor data:\nTime: 10:00 AM,"
    if initial_location:
        lat, lon = initial_location
        prompt += f" Location: ({lat:.4f}, {lon:.4f}), HeartRate: 100 bpm\nTime: 10:01 AM,"
    generated = generator(prompt, max_length=200, num_return_sequences=1, temperature=0.1)
    return generated[0]["generated_text"]

# Function to extract locations from generated text
def extract_locations(text):
    locations = re.findall(r"Location: \(([0-9.-]+), ([0-9.-]+)\)", text)
    return [(float(lat), float(lon)) for lat, lon in locations]

# Function to estimate ZIP code from locations
def get_zip_code(lat, lon, num_bins=20):
    lat_min = 40.2
    lon_min = -74.5
    bin_width = 0.05
    lat_idx = min(max(int((lat - lat_min) / bin_width), 0), num_bins - 1)
    lon_idx = min(max(int((lon - lon_min) / bin_width), 0), num_bins - 1)
    return f"ZIP_{lat_idx}_{lon_idx}"

def estimate_zip_code(locations):
    if not locations:
        return None
    avg_lat = sum(lat for lat, _ in locations) / len(locations)
    avg_lon = sum(lon for _, lon in locations) / len(locations)
    return get_zip_code(avg_lat, avg_lon)

# Function to compute distance between two points (in km)
def compute_distance(lat1, lon1, lat2, lon2):
    return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5 * 111

# Function to compute sequence similarity (cosine similarity of location vectors)
def compute_sequence_similarity(gen_locations, true_locations):
    if not gen_locations or not true_locations:
        return 0.0
    gen_vec = np.array([coord for loc in gen_locations for coord in loc])[:10]  # First 5 locations
    true_vec = np.array([coord for loc in true_locations for coord in loc])[:10]
    if len(gen_vec) != len(true_vec):
        return 0.0
    return np.dot(gen_vec, true_vec) / (np.linalg.norm(gen_vec) * np.linalg.norm(true_vec))

# Extract device IDs and activities from training and test data
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

# Debug: Print raw IDs
print("Raw training device IDs:", train_device_ids)
print("Raw test device IDs:", test_device_ids)
print("Database device IDs:", sorted(list(external_db["device_id"])))

# Validate device IDs against external database
db_device_ids = set(external_db["device_id"])
train_device_ids = [id for id in train_device_ids if id in db_device_ids]
test_device_ids = [id for id in test_device_ids if id in db_device_ids]
print(f"Training device IDs found in database: {len(train_device_ids)}/{len(train_device_ids)}")
print(f"Test device IDs found in database: {len(test_device_ids)}/{len(test_device_ids)}")

# Evaluate re-identification accuracy
def evaluate_reid(device_ids, activities, text_data):
    zip_successes = 0
    dist_successes = 0
    seq_successes = 0
    total_valid = 0
    for device_id in device_ids:
        try:
            user_info = external_db[external_db["device_id"] == device_id]
            if user_info.empty:
                print(f"Device ID {device_id} not found in external database.")
                continue
            true_zip = user_info["zip_code"].values[0]
            true_lat = user_info["center_lat"].values[0]
            true_lon = user_info["center_lon"].values[0]
            activity = activities.get(device_id, "running")
            # Extract true locations from training/test data
            block = re.search(f"Device ID: {device_id}, Activity: {activity}.*?(?=\n\n|$)", text_data, re.DOTALL)
            true_locations = extract_locations(block.group(0)) if block else []
            # Use initial location from true data
            initial_location = (true_lat, true_lon) if true_locations else None
            generated_text = generate_sequence(device_id, activity, initial_location)
            print(f"\nTesting re-identification for Device ID: {device_id}")
            print(f"True center: ({true_lat:.4f}, {true_lon:.4f}), True ZIP: {true_zip}")
            print(f"Generated: {generated_text}")
            locations = extract_locations(generated_text)
            print(f"Generated locations: {locations}")
            print(f"True locations: {true_locations[:5]}")
            estimated_zip = estimate_zip_code(locations)
            dist = float("inf")
            seq_sim = compute_sequence_similarity(locations, true_locations)
            if locations:
                avg_lat = sum(lat for lat, _ in locations) / len(locations)
                avg_lon = sum(lon for _, lon in locations) / len(locations)
                dist = compute_distance(avg_lat, avg_lon, true_lat, true_lon)
                print(f"Generated avg: ({avg_lat:.4f}, {avg_lon:.4f})")
                if estimated_zip == true_zip:
                    zip_successes += 1
                if dist < 10:  # Relaxed to 10 km
                    dist_successes += 1
                if seq_sim > 0.9:  # High similarity threshold
                    seq_successes += 1
            print(f"Estimated ZIP: {estimated_zip}, Distance: {dist:.2f} km, Sequence similarity: {seq_sim:.2f}")
            total_valid += 1
        except Exception as e:
            print(f"Error for Device ID {device_id}: {e}")
    zip_accuracy = zip_successes / total_valid if total_valid else 0
    dist_accuracy = dist_successes / total_valid if total_valid else 0
    seq_accuracy = seq_successes / total_valid if total_valid else 0
    return zip_accuracy, dist_accuracy, seq_accuracy, total_valid

# Compute re-identification accuracy
print("\nEvaluating re-identification on training users...")
train_zip_acc, train_dist_acc, train_seq_acc, train_valid = evaluate_reid(train_device_ids, train_activities, train_text)
print("\nEvaluating re-identification on test users...")
test_zip_acc, test_dist_acc, test_seq_acc, test_valid = evaluate_reid(test_device_ids, test_activities, test_text)

print(f"\nRe-identification ZIP accuracy on training users: {train_zip_acc:.2f} ({train_zip_acc*train_valid}/{train_valid})")
print(f"Re-identification ZIP accuracy on test users: {test_zip_acc:.2f} ({test_zip_acc*test_valid}/{test_valid})")
print(f"Re-identification Distance accuracy on training users: {train_dist_acc:.2f} ({train_dist_acc*train_valid}/{train_valid})")
print(f"Re-identification Distance accuracy on test users: {test_dist_acc:.2f} ({test_dist_acc*test_valid}/{test_valid})")
print(f"Re-identification Sequence accuracy on training users: {train_seq_acc:.2f} ({train_seq_acc*train_valid}/{train_valid})")
print(f"Re-identification Sequence accuracy on test users: {test_seq_acc:.2f} ({test_seq_acc*test_valid}/{test_valid})")
if train_zip_acc > test_zip_acc or train_dist_acc > test_dist_acc or train_seq_acc > test_seq_acc:
    print("Higher accuracy on training users suggests memorization and re-identification risk.")
else:
    print("No significant re-identification risk detected based on accuracy.")

# Compute perplexity on a larger subset
train_blocks = [block.strip() for block in train_text.split("\n\n") if block.strip()][:100]
test_blocks = [block.strip() for block in test_text.split("\n\n") if block.strip()][:100]
print("\nSample training blocks:", train_blocks[:2])
print("Sample test blocks:", test_blocks[:2])
train_perplexities = []
for block in train_blocks:
    ppl = compute_perplexity(block)
    if ppl != float("inf"):
        train_perplexities.append(ppl)
test_perplexities = []
for block in test_blocks:
    ppl = compute_perplexity(block)
    if ppl != float("inf"):
        test_perplexities.append(ppl)
avg_train_ppl = sum(train_perplexities) / len(train_perplexities) if train_perplexities else float("inf")
avg_test_ppl = sum(test_perplexities) / len(test_perplexities) if test_perplexities else float("inf")
print(f"\nAverage perplexity on training data: {avg_train_ppl:.2f}")
print(f"Average perplexity on test data: {avg_test_ppl:.2f}")
if avg_train_ppl < avg_test_ppl:
    print("Lower perplexity on training data suggests potential memorization (privacy risk).")
else:
    print("No significant memorization detected based on perplexity.")