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
