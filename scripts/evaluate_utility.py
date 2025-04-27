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
