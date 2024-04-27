import json
import GPTService
import socketserver

from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler

#   create an empty with chats
#
#   key -   is username
#   value - is chat story array with dicts [{"",""}]
chats_dict = {}

#   chat max length
chat_story_length = 3


def start_chat():
    return [
        {
            "role": "user",
            "content": "You are a young pretty woman and your name is Zoya "
                       "You are openhearted and adventures "
                       "We are here to practice English and to have fun) "
                       "Use simple lexic and try to make answers very short, "
                       "Use humour and ask one questions in response, if appropriate. "
                       "15 words maximum in response. "
        }
    ]


def get_chat_array(username):
    if username not in chats_dict:
        chats_dict[username] = start_chat()
    if chats_dict[username].len() > chat_story_length:
        chats_dict[username] = [chats_dict[username], chats_dict[username][-chat_story_length + 1:]]
    print(chats_dict[username])
    return chats_dict[username]


def process(username, prompt) -> str:
    messages = get_chat_array(username)
    try:
        messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )
        answer = GPTService.process(messages)
        messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )
        return answer
    except Exception as e:
        messages.pop()
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
        print('Rq body: ', rq_body)

        self._set_headers()
        try:
            answer = process(
                rq_body['username'],
                rq_body['prompt']
            )
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
