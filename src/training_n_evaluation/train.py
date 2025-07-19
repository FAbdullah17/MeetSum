# train.py
import os
import gc
import zipfile
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoTokenizer, BartForConditionalGeneration,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)

# === Extract data ===
zip_path = "./tokenized_data1.zip"
extract_path = "./tokenized_data1"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# === Load dataset ===
train_dataset = load_from_disk("./tokenized_data1/tokenized_data/train")
val_dataset = load_from_disk("./tokenized_data1/tokenized_data/validation")
test_dataset = load_from_disk("./tokenized_data1/tokenized_data/test")

dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset,
})

# === Load model and tokenizer ===
model_name = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.model_max_length = 1024
model = BartForConditionalGeneration.from_pretrained(model_name)

# Optional: truncate inputs
def truncate_inputs(example):
    example["input_ids"] = example["input_ids"][:1024]
    example["attention_mask"] = example["attention_mask"][:1024]
    return example

dataset = dataset.map(truncate_inputs)

# === Define training arguments ===
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=2,
    learning_rate=3e-5,
    eval_strategy="steps",
    save_strategy="steps",
    logging_steps=50,
    save_steps=200,
    eval_steps=200,
    save_total_limit=2,
    fp16=True,
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# === Train ===
gc.collect()
trainer.train()

# === Save model ===
save_directory = "./models"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)