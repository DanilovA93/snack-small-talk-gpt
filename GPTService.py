from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
access_token = "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"
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=access_token,
    torch_dtype=torch.float16,
    quantization_config=quantization_config
)


def process(
        chat,
        max_new_tokens=128,
        temperature=0.9,
        top_p=0.9,
        top_k=40,
) -> str:

    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=1.2,
        do_sample=True,
        seed=42,
    )

    tokenized_chat = tokenizer.apply_chat_template(chat, return_tensors="pt")
    inputs = tokenized_chat.to(device)

    outputs = model.generate(
        inputs,
        **generate_kwargs,
        stream=False,
        details=True,
        return_full_text=False
    )

    gen_answer = tokenizer.batch_decode(
        outputs[:, inputs.shape[1]:]
    )[0]

    answer = gen_answer[:-4]  # to remove </s>

    return answer
