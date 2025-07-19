# data_preprocessing.py
import re
from datasets import load_from_disk, DatasetDict

def clean_text(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    return text

def preprocess_function(example):
    transcript = clean_text(example["transcript"])
    summary = clean_text(example["summary"])
    if transcript and summary:
        return {"transcript": transcript, "summary": summary}
    else:
        return None

def preprocess_and_save():
    path = "C:/Users/Fahad's WorkStation/Programs/PF/MeetSum/data"
    meetingbank = DatasetDict({
        "train": load_from_disk(f"{path}/train"),
        "validation": load_from_disk(f"{path}/validation"),
        "test": load_from_disk(f"{path}/test"),
    })

    for split in meetingbank:
        meetingbank[split] = meetingbank[split].filter(lambda x: x["transcript"] and x["summary"])
        meetingbank[split] = meetingbank[split].map(preprocess_function)

    meetingbank["train"].save_to_disk("data/cleaned/train")
    meetingbank["validation"].save_to_disk("data/cleaned/validation")
    meetingbank["test"].save_to_disk("data/cleaned/test")

    print("✅ Preprocessing complete and saved to data/cleaned/")

# Uncomment to test as script
# if __name__ == "__main__":
#     preprocess_and_save()
