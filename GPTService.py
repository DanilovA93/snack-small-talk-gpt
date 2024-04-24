from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_id = "mistralai/Mistral-7B-Instruct-v0.1"
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
        max_new_tokens=100,
        temperature=0.7,
        top_p=1.0,
        top_k=40,
) -> str:

    tokenized_chat = tokenizer.apply_chat_template(chat, return_tensors="pt")
    inputs = tokenized_chat.to(device)

    outputs = model.generate(
        inputs,

        #       integer or null >= 0
        #       Default: null
        #       The maximum number of tokens to generate in the completion.
        #
        #       The token count of your prompt plus max_new_tokens cannot exceed the model's context length.
        max_new_tokens=max_new_tokens,

        #       bool
        do_sample=True,

        #       number or null [ 0 .. 1 ]
        #       Default: 0.7
        #       What sampling temperature to use, between 0.0 and 1.0. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
        #
        #       We generally recommend altering this or top_p but not both.
        temperature=temperature,

        #       number or null [ 0 .. 1 ]
        #       Default: 1
        #       Nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
        #
        #       We generally recommend altering this or temperature but not both.
        top_p=top_p,

        #       number or null [ 0 .. 200 ]
        #       Default: 50
        #       Controls the number of most-likely candidates that the model considers for the next token.
        top_k=top_k
    )

    gen_answer = tokenizer.batch_decode(
        outputs[:, inputs.shape[1]:]
    )[0]

    answer = gen_answer[:-4]  # to remove </s>

    return answer
