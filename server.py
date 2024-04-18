import http.server
import socketserver
import json
from http import HTTPStatus
from transformers import AutoTokenizer, AutoModelForCausalLM


model_id = "CohereForAI/c4ai-command-r-plus"
access_token = "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"


tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

def generate(test_prompt) -> str:

    messages = [{"role": "user", "content": test_prompt}]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

    gen_tokens = model.generate(
        input_ids,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.3,
    )

    gen_text = tokenizer.decode(gen_tokens[0])

    return gen_text


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
