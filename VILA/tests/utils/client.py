from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000",
    api_key="fake-key",
)

# image test
response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://blog.logomyway.com/wp-content/uploads/2022/01/NVIDIA-logo.jpg",
                    },
                },
            ],
        }
    ],
    max_tokens=300,
    model="VILA1.5-3B",
    extra_body={"num_beams": 1, "use_cache": False},
)
print(response.choices[0].message.content)

# text only test
response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [{"type": "text", "text": "What are you capable of doing?"}],
        }
    ],
    max_tokens=300,
    model="VILA1.5-3B",
    extra_body={"num_beams": 1, "use_cache": False},
)
print(response.choices[0].message.content)

# ================================================================
# Stream test

response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://blog.logomyway.com/wp-content/uploads/2022/01/NVIDIA-logo.jpg",
                    },
                },
            ],
        }
    ],
    max_tokens=300,
    model="VILA1.5-3B",
    extra_body={"num_beams": 1, "use_cache": False},
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content, end="")

response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [{"type": "text", "text": "What are you capable of doing?"}],
        }
    ],
    max_tokens=300,
    model="VILA1.5-3B",
    extra_body={"num_beams": 1, "use_cache": False},
    stream=True,
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="")
