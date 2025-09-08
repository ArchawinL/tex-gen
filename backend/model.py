import os 
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


dotenv_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path)
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")



MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

print(f"Loading {MODEL_NAME}")



tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    token=HF_TOKEN
)


generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

