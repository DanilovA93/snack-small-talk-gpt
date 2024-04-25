import torch

from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
access_token = "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"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=access_token,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.unk_token

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=access_token,
    device_map="auto",
    quantization_config=quantization_config
)

generate_kwargs = dict(
    temperature=0.9,
    max_new_tokens=128,
    top_p=0.92,
    repetition_penalty=0.1,
    do_sample=True,
)

chatbot = pipeline(
    task="conversational",
    model=model,
    tokenizer=tokenizer,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.unk_token_id,
    **generate_kwargs
)


def process(chat) -> str:
    conversation = chatbot(chat)
    return conversation.messages[-1]["content"]

# INSTALL TENSERFLOW ??