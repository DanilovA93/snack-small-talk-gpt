import http.server
import socketserver
import json
from http import HTTPStatus
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
access_token = "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"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    disable_exllama=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=access_token,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
)


def generate(prompt) -> str:
    messages = [
        {
            "role": "user",
            "content": "hi, my name is Anton"
        },
        {
            "role": "assistant",
            "content": "Nice to meet you, my name is GPT"
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=False).to("cuda")
    outputs = model.generate(input_ids, max_new_tokens=128, do_sample=False)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


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
