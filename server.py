import http.server
import socketserver
import json
from http import HTTPStatus
from transformers import AutoModelForCausalLM, AutoTokenizer


model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit_fp32_cpu_offload=True, device_map="auto")


def generate(test_prompt) -> bytes:
    messages = [
        {"role": "user", "content": test_prompt},
   ]

    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

    outputs = model.generate(inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    return "hi"

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
        self.wfile.write(generate(text))

    def do_GET(self):
        self.send_response(HTTPStatus.OK)
        self.end_headers()


httpd = socketserver.TCPServer(('', 8001), Handler)
httpd.serve_forever()
