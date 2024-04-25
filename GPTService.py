# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
#
# torch.random.manual_seed(0)
#
# print("model...")
# model = AutoModelForCausalLM.from_pretrained(
#     "microsoft/Phi-3-mini-128k-instruct",
#     device_map="cuda",
#     torch_dtype="auto",
#     trust_remote_code=True,
# )
#
# print("tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
#
# print("pipeline...")
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
# )
#
# print("generation_args...")
# generation_args = {
#     "max_new_tokens": 100,
#     "return_full_text": False,
#     "temperature": 0.0,
#     "do_sample": False,
# }
#
# print("ready")
# def process(chat) -> str:
#     print("process...")
    # output = pipe(chat, **generation_args)
    # print(output[0]['generated_text'])
    # return "hello"
