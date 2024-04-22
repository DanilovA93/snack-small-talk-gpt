import http.server
import socketserver
import json
from http import HTTPStatus
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
access_token = "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"
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
chat_story_length = 30

#   create an empty with chats
#
#   key -   is username
#   value - is chat story array with dicts [{"",""}]
chats_dict = {}

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=access_token,
    torch_dtype=torch.float16,
    quantization_config=quantization_config
)


def start_chat():
    return [
        {
            "role": "user",
            "content": "Pretend you are a young pretty woman"
        },
        {
            "role": "assistant",
            "content": "Okey, i am a young and pretty woman and my name is Zoya"
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
            "content": "Okey-dokey, 15 words maximum"
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


def process(
        username,
        prompt,
        max_new_tokens=100,
        temperature=0.7,
        top_p=1.0,
        top_k=40,
) -> str:
    messages = [
        {
            "role": "user",
            "content": "Your role is a young pretty openhearted woman. "
                       "I want to practice English and to have fun) "
                       "We don't use emoji. "
                       "Your answers must be short, 20 words maximum. "
                       "If you ask a question, ask only one question. "
                       "Use simple lexic and humour if appropriate. "
        },
        {
            "role": "assistant",
            "content": "Okey"
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    # messages = get_chat_array(username)
    #
    # messages.append(
    #     {
    #         "role": "user",
    #         "content": prompt
    #     }
    # )

    tokenized_chat = tokenizer.apply_chat_template(messages, return_tensors="pt")
    inputs = tokenized_chat.to(device)

    outputs = model.generate(
        inputs,

        #       integer or null >= 0
        #       Default: null
        #       The maximum number of tokens to generate in the completion.
        #
        #       The token count of your prompt plus max_new_tokens cannot exceed the model's context length.
        max_new_tokens=max_new_tokens,

        #       bool
        do_sample=False,

        #       number or null [ 0 .. 1 ]
        #       Default: 0.7
        #       What sampling temperature to use, between 0.0 and 1.0. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
        #
        #       We generally recommend altering this or top_p but not both.
        temperature=temperature,

        #       number or null [ 0 .. 1 ]
        #       Default: 1
        #       Nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
        #
        #       We generally recommend altering this or temperature but not both.
        top_p=top_p,

        #       number or null [ 0 .. 200 ]
        #       Default: 50
        #       Controls the number of most-likely candidates that the model considers for the next token.
        top_k=top_k
    )

    gen_answer = tokenizer.batch_decode(
        outputs[:, inputs.shape[1]:]
    )[0]

    answer = gen_answer[:-4]  # to remove </s>

    # messages.append(
    #     {
    #         "role": "assistant",
    #         "content": answer
    #     }
    # )

    return answer


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
                rq_body['prompt'],
                rq_body.get("max_new_tokens", None),
                rq_body.get("temperature", None),
                rq_body.get("top_p", None),
                rq_body.get("top_k", None)
            )
            self.wfile.write(answer.encode())
        except KeyError as err:
            self.wfile.write(f"Ошибка, отсутствуют необходимые параметры в теле запроса: {err}".encode())

    def do_GET(self):
        self.send_response(HTTPStatus.OK)
        self.end_headers()


httpd = socketserver.TCPServer(('', 8001), Handler)
httpd.serve_forever()
