import http.server
import socketserver
import json
from http import HTTPStatus
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


print(1)
model_mistral = AutoModelForCausalLM.from_pretrained(
    "alokabhishek/Mistral-7B-Instruct-v0.2-GGUF",
    model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf", # replace Q4_K_M.gguf with Q5_K_M.gguf as needed
    model_type="mistral",
    gpu_layers=50, # Use `gpu_layers` to specify how many layers will be offloaded to the GPU.
    hf=True
)
print(2)
tokenizer_mistral = AutoTokenizer.from_pretrained(
    "alokabhishek/Mistral-7B-Instruct-v0.2-GGUF", use_fast=True
)
print(3)
pipe_mistral = pipeline(model=model_mistral, tokenizer=tokenizer_mistral, task='text-generation')

# tokenizer.save_pretrained('./Mistral-7B-Instruct-v0.2-GGUF')


def generate(test_prompt) -> str:
    output_mistral = pipe_mistral(test_prompt, max_new_tokens=512)

    return output_mistral[0]["generated_text"]


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

#
#
# import http.server
# import socketserver
# import json
# from http import HTTPStatus
# from llama_cpp import Llama
#
#
# system_prompt = "You are a teacher."
# model_path = "./mistral-7b-instruct-v0.2.Q4_K_M.gguf"
#
# # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
# llm = Llama(
#     model_path=model_path,  # Download the model file first
#     chat_format="llama-2",
#     n_gpu_layers=35,        # The number of layers to offload to GPU, if you have GPU acceleration available
#     n_ctx=32768,            # The max sequence length to use - note that longer sequence lengths require much more resources
#     n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
# )
#
#
# def generate(test_prompt) -> str:
#
#     gpt = llm.create_chat_completion(
#         max_tokens=50,
#         temperature=0.0,
#         messages=[
#             {
#                 "role": "system",
#                 "content": system_prompt
#             },
#             {
#                 "role": "user",
#                 "content": test_prompt
#             }
#         ]
#     )
#
#     # gpt = llm(
#     #     "<s>[INST] {prompt} [/INST]", # Prompt
#     #     max_tokens=20,  # Generate up to 512 tokens
#     #     stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
#     #     echo=True        # Whether to echo the prompt
#     # )
#
#     print(gpt)
#
#     return gpt["choices"][0]["message"]["content"]
#
#
# class Handler(http.server.SimpleHTTPRequestHandler):
#     def _set_headers(self):
#         self.send_response(HTTPStatus.OK)
#         self.send_header('Content-type', 'text/plain')
#         # Allow requests from any origin, so CORS policies don't
#         # prevent local development.
#         self.send_header('Access-Control-Allow-Origin', '*')
#         self.end_headers()
#
#     def do_POST(self):
#         content_len = int(self.headers.get('Content-Length'))
#         message = json.loads(self.rfile.read(content_len))
#         text = message['prompt']
#         self._set_headers()
#         self.wfile.write(generate(text).encode())
#
#     def do_GET(self):
#         self.send_response(HTTPStatus.OK)
#         self.end_headers()
#
#
# httpd = socketserver.TCPServer(('', 8001), Handler)
# httpd.serve_forever()
