import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu
torch.manual_seed(100)
import time

MAX_NUM_FRAMES =64
frame_gap = 1
def encode_image(image):
    if not isinstance(image, Image.Image):
        if isinstance(image, str):
            # 直接是文件路径字符串
            image = Image.open(image).convert("RGB")
        elif hasattr(image, 'path'):
            image = Image.open(image.path).convert("RGB")
        else:
            image = Image.open(image.file.path).convert("RGB")
    # resize to max_size
    max_size = 448*16 
    if max(image.size) > max_size:
        w,h = image.size
        if w > h:
            new_w = max_size
            new_h = int(h * max_size / w)
        else:
            new_h = max_size
            new_w = int(w * max_size / h)
        image = image.resize((new_w, new_h), resample=Image.BICUBIC)
    return image

def encode_video(video):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

        # 修复视频路径获取逻辑
    if isinstance(video, str):
        # 直接是文件路径字符串
        video_path = video
    elif hasattr(video, 'path'):
        # Gradio 上传的文件对象，有 path 属性
        video_path = video.path
    elif hasattr(video, 'file') and hasattr(video.file, 'path'):
        # 某些情况下的嵌套文件对象
        video_path = video.file.path
    else:
        # 尝试其他可能的属性
        video_path = str(video)
    vr = VideoReader(video_path, ctx=cpu(0))

    sample_fps = round(vr.get_avg_fps() / frame_gap)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx)>MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    video = vr.get_batch(frame_idx).asnumpy()
    video = [Image.fromarray(v.astype('uint8')) for v in video]
    video = [encode_image(v) for v in video]
    print('video frames:', len(video))
    return video


model_path = '/home/rhs/code_workspace/VLM/hf_models/MiniCPM-V-4_5'
model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
model = model.eval().to('cuda:1')
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True) # or openbmb/MiniCPM-o-2_6


video = '/home/rhs/code_workspace/VLM/MiniCPM-V/assets/demo_video.mp4'
enable_thinking=False # If `enable_thinking=True`, the long-thinking mode is enabled.

# First round chat 
params = {
    'sampling': True,
    'top_p': 0.8,
    'top_k': 100,
    'temperature': 0.7,
    'repetition_penalty': 1.05,
    "max_new_tokens": 2048,
    "max_inp_length": 4352,  # 视频需要更长的输入长度
    "max_slice_nums": 1,
    "use_image_id": False,
}


msgs = []
msgs.extend(encode_video(video))
msgs.append("请描述视频内容。")
context = [{
    'role': 'user', 
    'content': msgs
}]
print(context)
answer = model.chat(
    msgs=context,
    tokenizer=tokenizer,
    enable_thinking=enable_thinking,
    **params
)
print(answer)



# msgs = [{
#     'role': 'user', 
#     'content': [
#         "白衣服的女生在做什么？",
#         encode_image(image), 
#         ]
# }]
# answer = model.chat(
#     msgs=msgs,
#     tokenizer=tokenizer,
#     enable_thinking=enable_thinking,
#     **params
# )