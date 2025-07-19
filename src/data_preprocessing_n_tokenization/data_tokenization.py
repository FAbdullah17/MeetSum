# data_tokenization.py
from datasets import load_from_disk, DatasetDict
from transformers import LEDTokenizerFast

def tokenize_batch(example, tokenizer):
    model_inputs = tokenizer(
        example["transcript"],
        padding="max_length",
        truncation=True,
        max_length=16384,
        return_attention_mask=True,
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["summary"],
            padding="max_length",
            truncation=True,
            max_length=256
        )["input_ids"]

    model_inputs["labels"] = labels
    return model_inputs

def tokenize_and_save():
    cleaned_path = "C:/Users/Fahad's WorkStation/Programs/PF/MeetSum/data/cleaned"
    dataset = DatasetDict({
        "train": load_from_disk(f"{cleaned_path}/train"),
        "validation": load_from_disk(f"{cleaned_path}/validation"),
        "test": load_from_disk(f"{cleaned_path}/test"),
    })

    tokenizer = LEDTokenizerFast.from_pretrained("allenai/led-base-16384")
    tokenizer.model_max_length = 16384

    tokenized_dataset = dataset.map(
        lambda x: tokenize_batch(x, tokenizer),
        batched=True,
        batch_size=1,
        num_proc=1
    )

    output_dir = "data/tokenized"
    tokenized_dataset.save_to_disk(output_dir)
    print("✅ Tokenization complete. Saved to:", output_dir)

# Uncomment to test as script
# if __name__ == "__main__":
#     tokenize_and_save()
