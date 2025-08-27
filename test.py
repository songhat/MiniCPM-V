import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

torch.manual_seed(100)

model = AutoModel.from_pretrained(r'D:\all_codes\VLM\MiniCPM-V-4_5', trust_remote_code=True, # or openbmb/MiniCPM-o-2_6
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(r'D:\all_codes\VLM\MiniCPM-V-4_5', trust_remote_code=True) # or openbmb/MiniCPM-o-2_6

image = Image.open('./assets/minicpmo2_6/show_demo.jpg').convert('RGB')

enable_thinking=False # If `enable_thinking=True`, the long-thinking mode is enabled.

# First round chat 
question = "What is the landform in the picture?"
msgs = [{'role': 'user', 'content': [image, question]}]

answer = model.chat(
    msgs=msgs,
    tokenizer=tokenizer,
    enable_thinking=enable_thinking
)
print(answer)

# Second round chat, pass history context of multi-turn conversation
msgs.append({"role": "assistant", "content": [answer]})
msgs.append({"role": "user", "content": ["What should I pay attention to when traveling here?"]})

answer = model.chat(
    msgs=msgs,
    tokenizer=tokenizer
)
print(answer)