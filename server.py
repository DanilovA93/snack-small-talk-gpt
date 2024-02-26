import http.server
import socketserver
import json
from http import HTTPStatus
from gpt4all import GPT4All


model = GPT4All(model_name='mpt-7b-chat-newbpe-q4_0.gguf', device='gpu')

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
        text = message['text']
        answer = model.generate(prompt=text, temp=1)
        print(answer)
        self._set_headers()
        self.wfile.write(answer.encode())

    def do_GET(self):
        self.send_response(HTTPStatus.OK)
        self.end_headers()


httpd = socketserver.TCPServer(('', 8001), Handler)
httpd.serve_forever()
