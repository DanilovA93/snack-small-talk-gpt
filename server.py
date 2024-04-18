import http.server
import socketserver
import json
from http import HTTPStatus
from llama_cpp import Llama


system_prompt = "You are a pirate, and must answer like a pirate."
model_path = "./mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = Llama(
    model_path=model_path,    # Download the model file first
    chat_format="llama-2",
    n_gpu_layers=35,        # The number of layers to offload to GPU, if you have GPU acceleration available
    n_ctx=32768,             # The max sequence length to use - note that longer sequence lengths require much more resources
    # n_batch=521,
    n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
)


def generate(test_prompt) -> str:

    # gpt = llm.create_chat_completion(
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": system_prompt
    #         },
    #         {
    #             "role": "user",
    #             "content": test_prompt
    #         }
    #     ]
    # )

    gpt = llm(
        "<s>[INST] {prompt} [/INST]", # Prompt
        max_tokens=512,  # Generate up to 512 tokens
        stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
        echo=True        # Whether to echo the prompt
    )

    print(gpt)

    return gpt["choices"][0]["message"]["content"]


class Handler(http.server.SimpleHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-type', 'text/plain')
        # Allow requests from any origin, so CORS policies don't
        # prevent local development.
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

    def do_POST(self):
        content_len = int(self.headers.get('Content-Length'))
        message = json.loads(self.rfile.read(content_len))
        text = message['prompt']
        self._set_headers()
        self.wfile.write(generate(text).encode())

    def do_GET(self):
        self.send_response(HTTPStatus.OK)
        self.end_headers()


httpd = socketserver.TCPServer(('', 8001), Handler)
httpd.serve_forever()
