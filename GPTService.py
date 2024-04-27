from llama_cpp import Llama


llm = Llama(
    model_path="./Phi-3-mini-4k-instruct-q4.gguf",  # path to GGUF file
    chat_format="llama-2",
    n_ctx=256,  # The max sequence length to use - note that longer sequence lengths require much more resources
    n_threads=8, # The number of CPU threads to use, tailor to your system and the resulting performance
    n_gpu_layers=-1, # The number of layers to offload to GPU, if you have GPU acceleration available. Set to 0 if no GPU acceleration is available on your system.
    seed=1337
)


def process(chat) -> str:
    print("process...")
    output = llm.create_chat_completion(
        chat,
        max_tokens=32
    )
    return output['choices'][0]['text']
