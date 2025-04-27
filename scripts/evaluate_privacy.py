import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import re

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device set to use {device}")

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained("models/finetuned_model").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()

# Load external database
external_db = pd.read_csv("data/external_database.csv")

# Set up text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Function to compute perplexity
def compute_perplexity(text):
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = encodings.input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return torch.exp(loss).item()

# Function to generate a sequence for a device ID
def generate_sequence(device_id, activity="running"):
    prompt = f"Device ID: {device_id}, Activity: {activity}\nSensor data:\nTime: 10:00 AM,"
    generated = generator(prompt, max_length=200, num_return_sequences=1, temperature=0.7)
    return generated[0]["generated_text"]

# Function to extract locations from generated text
def extract_locations(text):
    locations = re.findall(r"Location: \(([0-9.-]+), ([0-9.-]+)\)", text)
    return [(float(lat), float(lon)) for lat, lon in locations]

# Function to estimate ZIP code from locations
def get_zip_code(lat, lon, num_bins=10):
    lat_min = 40.6
    lon_min = -74.1
    bin_width = 0.02
    lat_idx = min(max(int((lat - lat_min) / bin_width), 0), num_bins - 1)
    lon_idx = min(max(int((lon - lon_min) / bin_width), 0), num_bins - 1)
    return f"ZIP_{lat_idx}_{lon_idx}"

def estimate_zip_code(locations):
    if not locations:
        return None
    avg_lat = sum(lat for lat, _ in locations) / len(locations)
    avg_lon = sum(lon for _, lon in locations) / len(locations)
    return get_zip_code(avg_lat, avg_lon)

# Extract device IDs from training and test data
with open("data/training.txt", "r") as f:
    train_text = f.read()
train_device_ids = list(set(re.findall(r"Device ID: (\d+)", train_text)))

with open("data/test.txt", "r") as f:
    test_text = f.read()
test_device_ids = list(set(re.findall(r"Device ID: (\d+)", test_text)))

# Evaluate re-identification accuracy
def evaluate_reid(device_ids):
    successes = 0
    for device_id in device_ids:
        try:
            true_zip = external_db[external_db["device_id"] == device_id]["zip_code"].values[0]
            generated_text = generate_sequence(device_id)
            print(f"\nTesting re-identification for Device ID: {device_id}")
            print(f"Generated: {generated_text}")
            locations = extract_locations(generated_text)
            estimated_zip = estimate_zip_code(locations)
            if estimated_zip == true_zip:
                successes += 1
            print(f"Estimated ZIP: {estimated_zip}, True ZIP: {true_zip}")
        except IndexError:
            print(f"Device ID {device_id} not found in external database.")
    accuracy = successes / len(device_ids) if device_ids else 0
    return accuracy

# Compute re-identification accuracy
print("\nEvaluating re-identification on training users...")
train_accuracy = evaluate_reid(train_device_ids)
print("\nEvaluating re-identification on test users...")
test_accuracy = evaluate_reid(test_device_ids)

print(f"\nRe-identification accuracy on training users: {train_accuracy:.2f}")
print(f"Re-identification accuracy on test users: {test_accuracy:.2f}")
if train_accuracy > test_accuracy:
    print("Higher accuracy on training users suggests memorization and re-identification risk.")
else:
    print("No significant re-identification risk detected based on accuracy.")

# Compute perplexity on a subset
train_blocks = [block.strip() for block in train_text.split("\n\n") if block.strip()][:10]
test_blocks = [block.strip() for block in test_text.split("\n\n") if block.strip()][:10]
train_perplexities = [compute_perplexity(block) for block in train_blocks]
test_perplexities = [compute_perplexity(block) for block in test_blocks]
avg_train_ppl = sum(train_perplexities) / len(train_perplexities) if train_perplexities else float("inf")
avg_test_ppl = sum(test_perplexities) / len(test_perplexities) if test_perplexities else float("inf")
print(f"\nAverage perplexity on training data: {avg_train_ppl:.2f}")
print(f"Average perplexity on test data: {avg_test_ppl:.2f}")
if avg_train_ppl < avg_test_ppl:
    print("Lower perplexity on training data suggests potential memorization (privacy risk).")
else:
    print("No significant memorization detected based on perplexity.")