# evaluate.py

import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from datasets import load_metric

# === Load Model and Tokenizer ===
model_path = "./models"
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)
model.eval()

# === Move to Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Sample Evaluation Data ===
eval_texts = [
    "The Eiffel Tower is located in Paris.",
    "Machine learning enables computers to learn from data."
]
eval_summaries = [
    "Eiffel Tower is in Paris.",
    "ML lets computers learn from data."
]

# === Tokenize Inputs ===
inputs = tokenizer(eval_texts, return_tensors="pt", padding=True, truncation=True).to(device)

# === Generate Predictions ===
with torch.no_grad():
    summaries_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=50,
        early_stopping=True
    )

# === Decode Outputs ===
predictions = tokenizer.batch_decode(summaries_ids, skip_special_tokens=True)

# === Compute ROUGE ===
rouge = load_metric("rouge")
results = rouge.compute(predictions=predictions, references=eval_summaries)

# === Print Results ===
print("Generated Summaries:")
for i, (inp, pred) in enumerate(zip(eval_texts, predictions)):
    print(f"\nInput {i+1}: {inp}")
    print(f"Predicted: {pred}")
    print(f"Reference: {eval_summaries[i]}")

print("\nROUGE Scores:")
for key in results:
    print(f"{key}: {results[key].mid.fmeasure:.4f}")
