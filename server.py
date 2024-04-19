import http.server
import socketserver
import json
from http import HTTPStatus
from transformers import AutoTokenizer, GemmaForCausalLM
import torch


model_id = "NexaAIDev/Octopus-v2"
access_token = "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"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = GemmaForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)


def generate(test_prompt) -> str:

    input_ids = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    input_length = input_ids["input_ids"].shape[1]
    outputs = model.generate(
        input_ids=input_ids["input_ids"],
        max_length=1024,
        do_sample=False
    )
    generated_sequence = outputs[:, input_length:].tolist()
    res = tokenizer.decode(generated_sequence[0])

    return res


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
