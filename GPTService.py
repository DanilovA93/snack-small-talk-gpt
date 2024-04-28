import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "microsoft/Phi-3-mini-128k-instruct"  # "mistralai/Mistral-7B-Instruct-v0.2"
access_token = "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"

# torch.random.manual_seed(0)
# device_count = torch.cuda.device_count()
# print(f"Device count = {device_count}")

print("Creating model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=access_token,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)#.half()
#model.config.pad_token_id = model.config.eos_token_id

print("Creating tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=access_token
)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

print("Building pipeline...")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

print("Generating args...")
generation_args = {
    "max_new_tokens": 75,
    "return_full_text": False,
    "temperature": 0.9,
    "do_sample": False,
}

print("GPT service is ready")


def process(chat) -> str:
    output = pipe(
        chat,
        batch_size=8,
        **generation_args)
    return output[0]['generated_text']
