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
