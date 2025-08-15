import os
from huggingface_hub import InferenceClient

#setx HF_API_KEY "your_Huggingface_access_token"

client = InferenceClient(
    provider="featherless-ai",
    api_key=os.environ["HF_API_KEY"],
)

completion = client.chat.completions.create(
    model="Intelligent-Internet/II-Medical-8B-1706",
    messages=[
        {
            "role": "user",
            "content": "I have a cough from 3 days"
        }
    ],
)

print(completion.choices[0].message.content)
