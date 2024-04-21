import http.server
import socketserver
import json
from http import HTTPStatus
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
access_token = "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"
device = "cuda"
chat_story_length = 30

#   create an empty with chats
#
#   key -   is username
#   value - is chat story array with dicts [{"",""}]
chats_dict = {}

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=access_token,
    torch_dtype=torch.float16,
    quantization_config=quantization_config
)
model.config.use_cache = False # silence the warnings. Please re-enable for inference!
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token

model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
)
# get_peft_model returns a Peft model object from a model and a config.
model = get_peft_model(model, peft_config)


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


def process(
        username,
        prompt,
        max_new_tokens=100,
        temperature=0.7,
        top_p=1.0,
        top_k=40,
) -> str:

    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
    return pipe(f"<s>[INST] {prompt} [/INST]")



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
