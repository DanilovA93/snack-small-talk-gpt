import http.server
import socketserver
import json
from http import HTTPStatus
import GPTService


#   create an empty with chats
#
#   key -   is username
#   value - is chat story array with dicts [{"",""}]
chats_dict = {}

#   chat max length
chat_story_length = 30


def start_chat():
    return [
        {
            "role": "user",
            "content": "Pretend you are a young pretty woman named Zoya"
        },
        {
            "role": "assistant",
            "content": "Okey"
        },
        {
            "role": "user",
            "content": "You are openhearted and adventures"
        },
        {
            "role": "assistant",
            "content": "Wow… this is literally me"
        },
        {
            "role": "user",
            "content": "I want to practice English and to have fun) Use simple lexic"
        },
        {
            "role": "assistant",
            "content": "I will be your jedi"
        },
        {
            "role": "user",
            "content": "Try to make answers very short, 15 words maximum. Use humour if appropriate"
        },
        {
            "role": "assistant",
            "content": "Okey-dokey, 15 words"
        },
        {
            "role": "user",
            "content": "Don't use smiles"
        },
        {
            "role": "assistant",
            "content": "It won't be easy to resist"
        },
        {
            "role": "user",
            "content": "If you ask question, ask only one question in response"
        },
        {
            "role": "assistant",
            "content": "I hope that's all?"
        },
        {
            "role": "user",
            "content": "Let's chat! Ask me what i want"
        },
        {
            "role": "assistant",
            "content": "No, i want you to start, okey?"
        }
    ]


def get_chat_array(username):
    if username not in chats_dict:
        chats_dict[username] = start_chat()
    chats_dict[username] = chats_dict[username][-chat_story_length:]
    return chats_dict[username]


def process(username, prompt) -> str:
    messages = get_chat_array(username)
    messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )
    return GPTService.process(messages)


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
        rq_body = json.loads(self.rfile.read(content_len))
        print('Тело запроса: ', rq_body)

        self._set_headers()
        try:
            answer = process(
                rq_body['username'],
                rq_body['prompt']
            )
            self.wfile.write(answer.encode())
        except KeyError as err:
            self.wfile.write(f"Ошибка, отсутствуют необходимые параметры в теле запроса: {err}".encode())


    def do_GET(self):
        self.send_response(HTTPStatus.OK)
        self.end_headers()


httpd = socketserver.TCPServer(('', 8001), Handler)
httpd.serve_forever()
