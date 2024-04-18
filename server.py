import http.server
import socketserver
import json
from http import HTTPStatus
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_id = "HuggingFaceH4/zephyr-7b-beta" #"mistralai/Mistral-7B-Instruct-v0.2"
# access_token = "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"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    disable_exllama=True
)

# tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     token=access_token,
#     quantization_config=bnb_config,
#     torch_dtype=torch.float16,
# )


def generate(prompt) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {
            "role": "user",
            "content": prompt
        },
    ]
    # tokenized_chat = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=True,
    #     add_generation_prompt=True,
    #     return_tensors="pt"
    # )
    #
    # return tokenizer.decode(tokenized_chat[0])

    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)



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
        self.wfile.write(generate(text).encode())

    def do_GET(self):
        self.send_response(HTTPStatus.OK)
        self.end_headers()


httpd = socketserver.TCPServer(('', 8001), Handler)
httpd.serve_forever()
