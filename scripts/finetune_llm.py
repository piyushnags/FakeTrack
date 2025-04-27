from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Prepare dataset for fine-tuning
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="data/training.txt",
    block_size=1024  # Length of input sequences
)
val_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="data/validation.txt",
    block_size=1024
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT-2 uses causal language modeling
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="models/finetuned_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="models/logs",
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=500,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Fine-tune
print("Starting fine-tuning on anonymized data...")
trainer.train()
trainer.save_model("models/finetuned_model")
print("Model saved to models/finetuned_model")
