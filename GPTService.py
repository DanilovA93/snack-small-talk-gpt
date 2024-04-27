import torch
import deepspeed
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.models.t5.modeling_t5 import T5Block

world_size = int(os.getenv('WORLD_SIZE', '1'))

model_id = "microsoft/Phi-3-mini-128k-instruct"

torch.random.manual_seed(0)

print("Creating model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
).half()
model.config.pad_token_id = model.config.eos_token_id

print("Creating tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

print("Building pipeline...")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)
pipe.model = deepspeed.init_inference(
    pipe.model,
    tensor_parallel={"tp_size": world_size},
    dtype=torch.float,
)

print("Generating args...")
generation_args = {
    "max_new_tokens": 75,
    "return_full_text": False,
    "temperature": 0.9,
    "do_sample": True,
}

print("GPT service is ready")


def process(chat) -> str:
    print("Processing...")
    output = pipe(chat, **generation_args)
    return output[0]['generated_text']
