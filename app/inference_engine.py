import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import os

# File parsers
from docx import Document
import PyPDF2

# === Load fine-tuned model and tokenizer ===
model_path = "./models"
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# === Read transcript file ===
def read_transcript_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == ".pdf":
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text

    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)

    else:
        raise ValueError("Unsupported file type. Please use .txt, .pdf, or .docx")

# === Chunk the input into token batches of <= 1024 ===
def chunk_input(text, tokenizer, max_tokens=1024):
    input_ids = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    chunks = [input_ids[i:i + max_tokens] for i in range(0, len(input_ids), max_tokens)]
    return chunks

# === Summarize one token chunk ===
def summarize_chunk(token_chunk):
    input_ids = token_chunk.unsqueeze(0).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            num_beams=4,
            max_length=128,
            early_stopping=True
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# === Full inference pipeline ===
def summarize_long_text(text):
    token_chunks = chunk_input(text, tokenizer)
    print(f"🧩 Total chunks: {len(token_chunks)}")
    summaries = []
    for i, chunk in enumerate(token_chunks):
        print(f"⏳ Processing chunk {i+1}/{len(token_chunks)}...")
        summaries.append(summarize_chunk(chunk))
    final_summary = " ".join(summaries)
    return final_summary

# === CLI usage ===
if __name__ == "__main__":
    print("📂 Enter path to transcript file (.txt, .pdf, .docx):")
    file_path = input(">> ").strip()

    if not os.path.exists(file_path):
        print("❌ File not found. Please check the path.")
        exit(1)

    try:
        transcript = read_transcript_file(file_path)
        if not transcript.strip():
            print("⚠️ The file is empty or unreadable.")
            exit(1)

        summary = summarize_long_text(transcript)
        print("\n=== FINAL SUMMARY ===\n")
        print(summary)

    except Exception as e:
        print(f"❌ Error: {e}")
