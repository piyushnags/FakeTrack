#!/bin/bash

# Exit on any error to ensure robustness
set -e

# Assume the script is run from the root of the existing FakeTrack repo
echo "Setting up project files in the existing FakeTrack repository..."

# Create subdirectories
mkdir -p data
mkdir -p models
mkdir -p scripts

# Create requirements.txt for dependencies
echo "Creating requirements.txt..."
cat <<EOL > requirements.txt
transformers[torch]
torch
faker
pandas
EOL

# Install dependencies (assumes Python and pip are installed)
echo "Installing dependencies..."
pip install -r requirements.txt --quiet

# Create generate_dataset.py to generate synthetic sensor data with PII
echo "Creating generate_dataset.py..."
cat <<EOL > scripts/generate_dataset.py
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
EOL

# Create finetune_llm.py to fine-tune GPT-2 on the synthetic dataset
echo "Creating finetune_llm.py..."
cat <<EOL > scripts/finetune_llm.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Prepare dataset for fine-tuning
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="data/training.txt",
    block_size=128  # Length of input sequences
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT-2 uses causal language modeling
)

# Define training arguments (optimized for quick testing)
training_args = TrainingArguments(
    output_dir="models/finetuned_model",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="models/logs",
    logging_steps=100,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Fine Mm-tune the model
print("Starting fine-tuning of GPT-2 model...")
trainer.train()
trainer.save_model("models/finetuned_model")
print("Model fine-tuned and saved to models/finetuned_model")
EOL

# Create evaluate_privacy.py to assess privacy via membership inference
echo "Creating evaluate_privacy.py..."
cat <<EOL > scripts/evaluate_privacy.py
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained("models/finetuned_model")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Function to compute perplexity for a given text
def compute_perplexity(text):
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return torch.exp(loss).item()

# Load training and test data
with open("data/training.txt", "r") as f:
    train_lines = [line.strip() for line in f.readlines()]
with open("data/test.txt", "r") as f:
    test_lines = [line.strip() for line in f.readlines()]

# Compute perplexity on a subset for speed
train_perplexities = [compute_perplexity(line) for line in train_lines[:10]]
test_perplexities = [compute_perplexity(line) for line in test_lines[:10]]

# Calculate average perplexities
avg_train_ppl = sum(train_perplexities) / len(train_perplexities)
avg_test_ppl = sum(test_perplexities) / len(test_perplexities)

# Output results
print(f"Average perplexity on training data: {avg_train_ppl:.2f}")
print(f"Average perplexity on test data: {avg_test_ppl:.2f}")
if avg_train_ppl < avg_test_ppl:
    print("Lower perplexity on training data suggests potential memorization (privacy risk).")
else:
    print("No significant memorization detected based on perplexity.")
EOL

# Create evaluate_utility.py to generate and assess synthetic data
echo "Creating evaluate_utility.py..."
cat <<EOL > scripts/evaluate_utility.py
from transformers import pipeline

# Load the fine-tuned model into a text generation pipeline
generator = pipeline("text-generation", model="models/finetuned_model", tokenizer="gpt2")

# Define a prompt to generate new sensor data
prompt = "User: John Doe, Time: 10:00 AM, Location:"

# Generate a sequence
print("Generating synthetic sensor data...")
generated = generator(prompt, max_length=50, num_return_sequences=1, temperature=0.7)
generated_text = generated[0]["generated_text"]

# Output the generated sequence
print("Generated sequence:")
print(generated_text)

# Basic validation
if "Location:" in generated_text and "HeartRate:" in generated_text:
    print("Utility check: Generated data contains expected fields (Location, HeartRate).")
else:
    print("Utility check: Generated data may be incomplete.")
EOL

# Execute each script in sequence
echo "Generating synthetic dataset..."
python scripts/generate_dataset.py

echo "Fine-tuning the LLM (this may take several minutes)..."
python scripts/finetune_llm.py

echo "Evaluating privacy..."
python scripts/evaluate_privacy.py

echo "Evaluating utility..."
python scripts/evaluate_utility.py

echo "FakeTrack project setup and execution completed successfully!"