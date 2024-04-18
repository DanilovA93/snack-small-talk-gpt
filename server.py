import http.server
import socketserver
import json
from http import HTTPStatus
import transformers
import torch


model_id = "meta-llama/Meta-Llama-3-8B"
access_token = "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"


pipeline = transformers.pipeline(
    "text-generation", model=model_id, token=access_token, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)


def generate(test_prompt) -> str:

    answer = pipeline(test_prompt)

    print(answer)


    return "=)"


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
