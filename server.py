import http.server
import socketserver
import json
from http import HTTPStatus
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


model_name = "deepset/roberta-base-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)



def generate(test_prompt) -> str:
    QA_input = {
        'question': test_prompt,
        'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
    }
    res = nlp(QA_input)
    print(res)
    return res["answer"]


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
