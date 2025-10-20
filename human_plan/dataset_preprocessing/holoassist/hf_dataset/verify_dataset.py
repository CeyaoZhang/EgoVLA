from datasets import load_from_disk

# Load your dataset (adjust the dataset loading as needed)
dataset = load_from_disk("data/ha_dataset/HoloAssist_HF_hands_filtered_train")  # Adjust 'your_dataset_name' and 'your_split'

def check_empty_entries(example):
    # Check if any field is None or an empty string
    for key, value in example.items():
        if value is None or value == "":
            return True  # Return True if any field is empty or None
    return False  # No empty or None fields

from tqdm import tqdm
# Loop over the dataset and log empty entries
empty_indices = []
for idx, entry in tqdm(enumerate(dataset)):
    if check_empty_entries(entry):
        print(entry["seq_name"], entry["frame_count"])
        empty_indices.append(idx)

# Output the indices of entries with empty fields
if empty_indices:
    print(f"Found {len(empty_indices)} empty entries at indices: {empty_indices}")
else:
    print("No empty entries found!")
