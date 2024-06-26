import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

cache_dict = {}

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
access_token = "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"

torch.random.manual_seed(0)

print("Creating model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=access_token,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
).half()
model.config.pad_token_id = model.config.eos_token_id

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
    batch_size=8
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
    last_request = chat[-1]["content"]
    cached_response = get_from_cache(last_request)
    if cached_response is not None:
        return cached_response
    else:
        output = pipe(chat, **generation_args)
        answer = output[0]['generated_text']
        cache(last_request, answer)
        return answer


def get_from_cache(request):
    request_hash = hash(request)
    key = str(request_hash)
    return cache_dict[key] if key in cache_dict else None


def cache(request, response):
    key = str(hash(request))
    cache_dict[key] = response
