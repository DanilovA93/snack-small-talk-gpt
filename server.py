import http.server
import socketserver
import json
from http import HTTPStatus
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


model_id = "mistralai/Mistral-7B-Instruct-v0.2"
access_token = "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"

tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
device = "cuda"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quanlst=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=access_token,
    torch_dtype=torch.float16,
    quantization_config=quantization_config
)


def generate(test_prompt) -> str:
    messages = [
        {
            "role": "user",
            "content": test_prompt
        }
    ]

    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    generated_ids = model.generate(
        inputs,
        max_tokens=50,          # Response text length.
        temperature=0.6,        # Ranges from 0 to 2, lower values ==> Determinism, Higher Values ==> Randomness
        top_p=1,                # Ranges 0 to 1. Controls the pool of tokens.  Lower ==> Narrower selection of words
        frequency_penalty=0,    # used to discourage the model from repeating the same words or phrases too frequently within the generated text
        presence_penalty=0,     # used to encourage the model to include a diverse range of tokens in the generated text.
        do_sample=True,
        prompt_template="<s>[INST] {prompt} [/INST] "
    )
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
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
