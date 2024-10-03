import torch
from transformers import pipeline
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

print(torch.version.cuda)

# パイプラインの準備
pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b-jpn-it",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda"
    )

# メッセージの準備
messages = [
    {"role": "user", "content": "ドラゴンボールでは誰が一番強いのか?"},
]

# 推論の実行
outputs = pipe(messages, return_full_text=False, max_new_tokens=1024)
assistant_response = outputs[0]["generated_text"].strip()
print(assistant_response)