from datasets import load_dataset
meetingbank = load_dataset("huuuyeah/meetingbank")

train_data = meetingbank['train']
test_data = meetingbank['test']
val_data = meetingbank['validation']

train_data.save_to_disk("data/raw/train")
val_data.save_to_disk("data/raw/validation")
test_data.save_to_disk("data/raw/test")