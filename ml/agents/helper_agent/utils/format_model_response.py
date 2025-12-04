import re

def format_model_response(response: str):
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    
    matches = list(re.finditer(r"<\|im_start\|>assistant", response))
    if not matches:
        return response.strip()
    last = matches[-1].start()

    return response[last + len("<|im_start|>assistant"):].strip()