from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" # the device to load the model onto
model_id = "Qwen/Qwen2-7B-Instruct"
access_token = "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"

print("Creating model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto"
)

print("Creating tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_id
)


def process(prompt) -> str:
    messages = [
        {"role": "system", "content": "Ты полезный помощник"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
