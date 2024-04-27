from llama_cpp import Llama


llm = Llama(
    model_path="./Phi-3-mini-4k-instruct-q4.gguf",  # path to GGUF file
    n_ctx=512,  # The max sequence length to use - note that longer sequence lengths require much more resources
    n_threads=8, # The number of CPU threads to use, tailor to your system and the resulting performance
    n_gpu_layers=35, # The number of layers to offload to GPU, if you have GPU acceleration available. Set to 0 if no GPU acceleration is available on your system.
)


def process(chat) -> str:
    print("process...")
    output = llm(
        chat,
        max_tokens=128,
        echo=True,  # Whether to echo the prompt
    )
    return output['choices'][0]['text']
