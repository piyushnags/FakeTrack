from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from dataclasses import dataclass
import pyrallis

@dataclass
class FinetuningConfig:
    # Model configuration
    model_name: str = "gpt2"
    block_size: int = 1024
    
    # Training configuration
    output_dir: str = "models/finetuned_model"
    learning_rate: float = 7e-4
    num_train_epochs: int = 20
    per_device_train_batch_size: int = 4
    save_steps: int = 10_000
    save_total_limit: int = 2
    logging_dir: str = "models/logs"
    logging_steps: int = 100
    eval_steps: int = 500
    
    # Data configuration
    train_file: str = "data/training.txt"
    val_file: str = "data/validation.txt"

@pyrallis.wrap()
def main(cfg: FinetuningConfig):
    # Load pre-trained GPT-2 model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(cfg.model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(cfg.model_name)

    # Prepare dataset for fine-tuning
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=cfg.train_file,
        block_size=cfg.block_size
    )
    val_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=cfg.val_file,
        block_size=cfg.block_size
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # GPT-2 uses causal language modeling
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        overwrite_output_dir=True,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        logging_dir=cfg.logging_dir,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
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
    trainer.save_model(cfg.output_dir)
    print(f"Model saved to {cfg.output_dir}")

if __name__ == "__main__":
    main()
