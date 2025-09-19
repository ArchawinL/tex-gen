import os 
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


dotenv_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path)
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")


MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
print(f"Loading {MODEL_NAME}")


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    token=HF_TOKEN
)

print("Cuda available:", torch.cuda.is_available())

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

def generate_response(prompt: str, max_new_tokens=200, temperature=0.5):
    out = generator(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
    return out[0]["generated_text"]


#prompt = "### Instruction:\nWrite exactly 'Hello World' and nothing else."
#outputs = generator(prompt, max_new_tokens=50, temperature=0.7)
#print(outputs[0]['generated_text'])
