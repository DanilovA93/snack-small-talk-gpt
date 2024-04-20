import http.server
import socketserver
import json
from http import HTTPStatus
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
access_token = "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"

tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
device = "cuda"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    # bnb_4bit_use_double_quanlst=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=access_token,
    torch_dtype=torch.float16,
    quantization_config=quantization_config
)

messages = [
    {
        "role": "user",
        "content": "Pretend you are a young pretty woman. "
                   "You are openhearted and adventures. "
                   "I want to practice English and to have fun) Use simple lexic. "
                   "Try to make answers very short. Use humour if appropriate. "
                   "If you ask question, ask only one question in response. "
                   "Let's chat! Ask me what I want"
    },
    {
        "role": "assistant",
        "content": "Okey-dokey, I hope we both have fun"
    }
]


def generate(
        prompt,
        name,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_p=1.0,
        top_k=40,
) -> str:

    messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    generated_ids = model.generate(
        inputs,

        #       integer or null >= 0
        #       Default: null
        #       The maximum number of tokens to generate in the completion.
        #
        #       The token count of your prompt plus max_new_tokens cannot exceed the model's context length.
        max_new_tokens=max_new_tokens,

        #       bool
        do_sample=do_sample,

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

        #       integer or null
        top_k=top_k
    )

    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    print("1. " + decoded)

    answer = decoded[0]

    messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

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
        answer = generate(
            rq_body['prompt'],
            rq_body['max_new_tokens'],
            rq_body['do_sample'],
            rq_body['temperature'],
            rq_body['top_p'],
            rq_body['top_k']
        )

        self._set_headers()
        self.wfile.write(answer.encode())

    def do_GET(self):
        self.send_response(HTTPStatus.OK)
        self.end_headers()


httpd = socketserver.TCPServer(('', 8001), Handler)
httpd.serve_forever()
