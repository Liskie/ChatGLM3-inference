# ChatGLM3 Inference

## Quickstart

1. Install the required packages
    ```shell
    pip install -r requirements.txt
    ``` 
    
2. Run the examples
    ```shell
    CUDA_VISIBLE_DEVICES="1,2" python main.py
    ```
    
## To use in your code

In `main.py` we provided two simple examples for single and multi turn inference, as shown below:

### Single Turn
```python
tokenizer, model = load_model(model_path='/data/private/wyj2021/models/chatglm3-6b', num_gpus=2)

response, _ = model.chat(tokenizer, "你好", history=[])
print(response)
```

### Multi Turn
```python
tokenizer, model = load_model(model_path='/data/private/wyj2021/models/chatglm3-6b', num_gpus=2)

response, history = model.chat(tokenizer, "你好", history=[])
print(response)
# Just pass the history to the next turn
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)
```