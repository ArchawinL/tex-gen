from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

print(f"Loading {MODEL_NAME}")


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)


generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

