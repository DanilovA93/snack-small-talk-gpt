import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-8B"
access_token = "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"

pipeline = transformers.pipeline(
    "text-generation",
    token=access_token,
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)


def process(chat) -> str:
    answer = pipeline(chat[-1])
    print(answer)
    return answer
