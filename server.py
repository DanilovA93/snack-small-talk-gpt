import http.server
import socketserver
import json
from http import HTTPStatus
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


model_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
)


def generate(test_prompt) -> str:
    messages = [
        {
            "role": "user",
            "content": test_prompt
        }
    ]

    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
    generated_ids = model.generate(inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    # outputs = model.generate(
    #     inputs,
    #     max_new_tokens=50,
    #     do_sample=True,
    #     temperature=0.1
    # )
    return decoded[0]


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
        prompt = message['prompt']
        self._set_headers()
        self.wfile.write(generate(prompt).encode())

    def do_GET(self):
        self.send_response(HTTPStatus.OK)
        self.end_headers()


httpd = socketserver.TCPServer(('', 8001), Handler)
httpd.serve_forever()
