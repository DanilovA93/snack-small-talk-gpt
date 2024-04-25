import torch
import threading

from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline

device_map = {"": 1}

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
access_token = "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"

print("Creating tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=access_token,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.unk_token

print("Creating quantization config...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print(f"Creating model {model_id}...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=access_token,
    device_map=device_map, #"auto",
    quantization_config=quantization_config
)

print("Creating generate kwargs...")
generate_kwargs = dict(
    temperature=0.9,
    max_new_tokens=128,
    top_p=0.92,
    repetition_penalty=1.0,
    do_sample=True,
)

print("Creating chatbot...")
chatbot = pipeline(
    task="conversational",
    model=model,
    tokenizer=tokenizer,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.unk_token_id,
    use_fast=True,
    **generate_kwargs
)


def process(chat) -> str:
    print(f"Processing on {threading.currentThread().name}...")
    conversation = chatbot(chat)
    return conversation.messages[-1]["content"]
