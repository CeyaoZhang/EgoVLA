from transformers import AutoTokenizer

# Load the tokenizer
tokenizer_name = "liuhaotian/llava-v1.5-7b"  # You can replace this with an
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Get the vocabulary
vocab = tokenizer.get_vocab()

# Sort the vocabulary by index
sorted_vocab = dict(sorted(vocab.items(), key=lambda item: item[1]))

# Print the sorted vocabulary
for token, index in sorted_vocab.items():
    print(f"{index}: {token}")

# Save the sorted vocabulary to a file
with open("sorted_vocabulary_by_index.txt", "w") as f:
    for token, index in sorted_vocab.items():
        f.write(f"{index}: {token}\n")

print("Sorted vocabulary by index has been saved to sorted_vocabulary_by_index.txt")
