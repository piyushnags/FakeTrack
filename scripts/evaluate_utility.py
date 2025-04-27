from transformers import pipeline

# Load the fine-tuned model
generator = pipeline("text-generation", model="models/finetuned_model", tokenizer="gpt2-large")

# Example prompt
prompt = (
    "Device ID: 0001, Activity: running\n"
    "Sensor data:\n"
    "Time: 10:00 AM, Location: 40.7128, -74.0060, HeartRate: 150 bpm\n"
    "Time: 10:01 AM,"
)

# Generate continuation
print("Generating sensor data...")
generated = generator(prompt, max_length=200, num_return_sequences=1, temperature=0.7)
generated_text = generated[0]["generated_text"]

print("Generated Output:")
print(generated_text)

# Utility checks
if "Location:" in generated_text and "HeartRate:" in generated_text:
    print("Utility Check: Contains expected fields (Location, HeartRate).")
    # Extract heart rate for range check
    try:
        hr_str = generated_text.split("HeartRate: ")[1].split()[0]
        hr = int(hr_str)
        if 60 <= hr <= 200:
            print("Utility Check: Heart rate within plausible range (60-200 bpm).")
        else:
            print("Utility Check: Heart rate out of plausible range.")
    except:
        print("Utility Check: Could not parse heart rate.")
else:
    print("Utility Check: Missing expected fields.")