import json
import GPTService
import socketserver

from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler


def process(system_prompt, user_prompt) -> str:
    try:
        answer = GPTService.process(system_prompt, user_prompt)
        return answer
    except Exception as e:
        raise Exception(e)


class Handler(SimpleHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-type', 'text/plain')
        # Allow requests from any origin, so CORS policies don't
        # prevent local development.
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

    def do_POST(self):
        content_len = int(self.headers.get('Content-Length'))
        rq_body = json.loads(self.rfile.read(content_len))
        print(f'Rq body: {rq_body}')

        self._set_headers()
        try:
            answer = process(rq_body['system_prompt'], rq_body['user_prompt'])
            print(f'Rs body: {answer}')
            self.wfile.write(answer.encode())
        except KeyError as err:
            self.wfile.write(f"Error, required parameters are missing in the request body: {err}".encode())
        except Exception as err:
            message = f"Error: {err}"
            print(message)
            self.wfile.write(message.encode())

    def do_GET(self):
        self.send_response(HTTPStatus.OK)
        self.end_headers()


httpd = socketserver.TCPServer(('', 8001), Handler)
httpd.serve_forever()
print("Server started, use <Ctrl-C> to stop")
