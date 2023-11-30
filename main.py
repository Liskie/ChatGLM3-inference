from typing import Any

from torch.nn import Module
from transformers import AutoTokenizer, AutoModel

from utils import load_model_on_gpus


def load_model(model_path: str = 'THUDM/chatglm3-6b', num_gpus: int = 1) -> tuple[Any, Module]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    match num_gpus:
        case 1:
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device='cuda')
        case _ if num_gpus >= 2:
            model = load_model_on_gpus(model_path, num_gpus=num_gpus)
        case _:
            raise ValueError("num_gpus must be a positive integer!")
    return tokenizer, model.eval()


if __name__ == '__main__':
    tokenizer, model = load_model(model_path='/data/private/wyj2021/models/chatglm3-6b', num_gpus=2)

    # If you only need on turn of conversation, you can use the following code:
    response, _ = model.chat(tokenizer, "你好", history=[])
    print(response)

    # If you need multiple turns of conversation, you can use the following code:
    response, history = model.chat(tokenizer, "你好", history=[])
    print(response)
    response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
    print(response)
