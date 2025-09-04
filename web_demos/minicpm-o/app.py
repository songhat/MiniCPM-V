#!/usr/bin/env python
# encoding: utf-8
import torch
import argparse
from transformers import AutoModel, AutoTokenizer
import gradio as gr
from PIL import Image
from decord import VideoReader, cpu
import io
import os
import copy
import requests
import base64
import json
import traceback
import re
import modelscope_studio as mgr
import uvicorn

# README, How to run demo on different devices

# For Nvidia GPUs.
# python chatbot_web_demo_o2.6.py


# Argparser
parser = argparse.ArgumentParser(description='demo')
parser.add_argument('--model', type=str , default=r"/home/rhs/code_workspace/VLM/hf_models/MiniCPM-V-4_5", help="huggingface model name or local path")
parser.add_argument('--multi-gpus', action='store_true', default=False, help='use multi-gpus')
args = parser.parse_args()
device = "cuda:1"
model_name = 'MMiniCPM-V-4_5'

# Load model
model_path = args.model
if args.multi_gpus:
    from accelerate import load_checkpoint_and_dispatch, init_empty_weights, infer_auto_device_map
    with init_empty_weights():
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, attn_implementation='sdpa', torch_dtype=torch.bfloat16,
            init_audio=False, init_tts=False)
    device_map = infer_auto_device_map(model, max_memory={0: "10GB", 1: "10GB"},
        no_split_module_classes=['SiglipVisionTransformer', 'Qwen2DecoderLayer'])
    device_id = device_map["llm.model.embed_tokens"]
    device_map["llm.lm_head"] = device_id # firtt and last layer should be in same device
    device_map["vpm"] = device_id
    device_map["resampler"] = device_id
    device_id2 = device_map["llm.model.layers.26"]
    device_map["llm.model.layers.8"] = device_id2
    device_map["llm.model.layers.9"] = device_id2
    device_map["llm.model.layers.10"] = device_id2
    device_map["llm.model.layers.11"] = device_id2
    device_map["llm.model.layers.12"] = device_id2
    device_map["llm.model.layers.13"] = device_id2
    device_map["llm.model.layers.14"] = device_id2
    device_map["llm.model.layers.15"] = device_id2
    device_map["llm.model.layers.16"] = device_id2
    #print(device_map)

    model = load_checkpoint_and_dispatch(model, model_path, dtype=torch.bfloat16, device_map=device_map)
else:
    # model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, init_audio=False, init_tts=False)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = model.to(device=device)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.eval()




ERROR_MSG = "Error, please retry"
MAX_NUM_FRAMES = 64
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.mov', '.avi', '.flv', '.wmv', '.webm', '.m4v'}

def get_file_extension(filename):
    return os.path.splitext(filename)[1].lower()

def is_image(filename):
    return get_file_extension(filename) in IMAGE_EXTENSIONS

def is_video(filename):
    return get_file_extension(filename) in VIDEO_EXTENSIONS


form_radio = {
    'choices': ['Beam Search', 'Sampling'],
    #'value': 'Beam Search',
    'value': 'Sampling',
    'interactive': True,
    'label': 'Decode Type'
}

memory_radio = {
    'choices': ['Long Memory', 'Short Memory'],
    'value': 'Long Memory',
    'interactive': True,
    'label': 'Memory Type'
}



def create_component(params, comp='Slider'):
    if comp == 'Slider':
        return gr.Slider(
            minimum=params['minimum'],
            maximum=params['maximum'],
            value=params['value'],
            step=params['step'],
            interactive=params['interactive'],
            label=params['label']
        )
    elif comp == 'Radio':
        return gr.Radio(
            choices=params['choices'],
            value=params['value'],
            interactive=params['interactive'],
            label=params['label']
        )
    elif comp == 'Button':
        return gr.Button(
            value=params['value'],
            interactive=True
        )


def create_multimodal_input(upload_image_disabled=False, upload_video_disabled=False):
    return mgr.MultimodalInput(
        upload_image_button_props={'label': 'Upload Image', 'disabled': upload_image_disabled, 'file_count': 'multiple'},
        upload_video_button_props={'visible': False},
        submit_button_props={'label': 'Submit'}
    )


def chat(img, msgs, ctx, params=None, vision_hidden_states=None):
    try:
        print('msgs:', msgs)
        answer = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            **params
        )
        res = re.sub(r'(<box>.*</box>)', '', answer)
        res = res.replace('<ref>', '')
        res = res.replace('</ref>', '')
        res = res.replace('<box>', '')
        answer = res.replace('</box>', '')
        print('answer:', answer)
        return 0, answer, None, None
    except Exception as e:
        print(e)
        traceback.print_exc()
        return -1, ERROR_MSG, None, None


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
    ## save by BytesIO and convert to base64
    #buffered = io.BytesIO()
    #image.save(buffered, format="png")
    #im_b64 = base64.b64encode(buffered.getvalue()).decode()
    #return {"type": "image", "pairs": im_b64}


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
    
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx)>MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    video = vr.get_batch(frame_idx).asnumpy()
    video = [Image.fromarray(v.astype('uint8')) for v in video]
    video = [encode_image(v) for v in video]
    print('video frames:', len(video))
    return video


def check_mm_type(mm_file):
    if hasattr(mm_file, 'path'):
        path = mm_file.path
    else:
        path = mm_file.file.path
    if is_image(path):
        return "image"
    if is_video(path):
        return "video"
    return None


def encode_mm_file(mm_file):
    if check_mm_type(mm_file) == 'image':
        return [encode_image(mm_file)]
    if check_mm_type(mm_file) == 'video':
        return encode_video(mm_file)
    return None

def make_text(text):
    #return {"type": "text", "pairs": text} # # For remote call
    return text

def encode_message(_question):
    files = _question.files
    question = _question.text
    pattern = r"\[mm_media\]\d+\[/mm_media\]"
    matches = re.split(pattern, question)
    message = []
    if len(matches) != len(files) + 1:
        gr.Warning("Number of Images not match the placeholder in text, please refresh the page to restart!")
    assert len(matches) == len(files) + 1

    text = matches[0].strip()
    if text:
        message.append(make_text(text))
    for i in range(len(files)):
        message += encode_mm_file(files[i])
        text = matches[i + 1].strip()
        if text:
            message.append(make_text(text))
    return message


def check_has_videos(_question):
    images_cnt = 0
    videos_cnt = 0
    for file in _question.files:
        if check_mm_type(file) == "image":
            images_cnt += 1 
        else:
            videos_cnt += 1
    return images_cnt, videos_cnt 


def count_video_frames(_context):
    num_frames = 0
    for message in _context:
        for item in message["content"]:
            #if item["type"] == "image": # For remote call
            if isinstance(item, Image.Image):
                num_frames += 1
    return num_frames


def respond(_question, _chat_bot, _app_cfg):
    _context = _app_cfg['ctx'].copy()
    print("短记忆模式")
    _context = _context[:2] + [{'role': 'user', 'content': encode_message(_question)}]

    images_cnt = _app_cfg['images_cnt']
    videos_cnt = _app_cfg['videos_cnt']
    files_cnts = check_has_videos(_question)
    if files_cnts[1] + videos_cnt > 1 or (files_cnts[1] + videos_cnt == 1 and files_cnts[0] + images_cnt > 0):
        gr.Warning("Only supports single video file input right now!")
        return _question, _chat_bot, _app_cfg

    params = {
        'sampling': True,
        'top_p': 0.8,
        'top_k': 100,
        'temperature': 0.7,
        'repetition_penalty': 1.05,
        "max_new_tokens": 2048
    }
    
    if files_cnts[1] + videos_cnt > 0:
        params["max_inp_length"] = 4352 # 4096+256
        params["use_image_id"] = False
        # 如果视频帧数大于16，则图片不进行slice，否则进行2片slice
        params["max_slice_nums"] = 1 if count_video_frames(_context) > 16 else 2

    code, _answer, _, sts = chat("", _context, None, params)

    images_cnt += files_cnts[0]
    videos_cnt += files_cnts[1]
   
    _context = _context[:3] + [{'role': 'user', 'content': encode_message(_question)}]
    _chat_bot.append((_question, _answer))

    if code == 0:
        _app_cfg['ctx']=_context
        _app_cfg['sts']=sts
    _app_cfg['images_cnt'] = images_cnt
    _app_cfg['videos_cnt'] = videos_cnt

    upload_image_disabled = videos_cnt > 0
    upload_video_disabled = videos_cnt > 0 or images_cnt > 0
    return create_multimodal_input(upload_image_disabled, upload_video_disabled), _chat_bot, _app_cfg

def chat_with_image(image, text, history):
    if image is None and not text.strip():
        gr.Warning("请上传图片或输入文字")
        return history, ""
    
    try:
        # 构建消息
        message = []
        if text.strip():
            message.append(make_text(text))
        if image is not None:
            encoded_image = encode_image(image)
            message.append(encoded_image)
        
        context = [{'role': 'user', 'content': message}]
        
        params = {
            'sampling': True,
            'top_p': 0.8,
            'top_k': 100,
            'temperature': 0.7,
            'repetition_penalty': 1.05,
            "max_new_tokens": 2048,
            # "max_slice_nums": 2,
            # "enable_thinking": True
        }
        
        code, answer, _, _ = chat("", context, None, params)
        
        if code == 0:
            user_message = text if text.strip() else "上传了一张图片"
            history.append((user_message, answer))
            return history, ""
        else:
            gr.Warning("生成回答时出错")
            return history, text
            
    except Exception as e:
        print(f"聊天错误: {e}")
        gr.Warning(f"处理失败: {str(e)}")
        return history, text

def clear_chat():
    return [], ""

def regenerate_button_clicked(_question, _chat_bot, _app_cfg):
    if len(_chat_bot) <= 1 or not _chat_bot[-1][1]:
        gr.Warning('No question for regeneration.')
        return '', _chat_bot, _app_cfg
    if _app_cfg["chat_type"] == "Chat":
        images_cnt = _app_cfg['images_cnt']
        videos_cnt = _app_cfg['videos_cnt']
        _question = _chat_bot[-1][0]
        _chat_bot = _chat_bot[:-1]
        _app_cfg['ctx'] = _app_cfg['ctx'][:-2]
        files_cnts = check_has_videos(_question)
        images_cnt -= files_cnts[0]
        videos_cnt -= files_cnts[1]
        _app_cfg['images_cnt'] = images_cnt
        _app_cfg['videos_cnt'] = videos_cnt
        upload_image_disabled = videos_cnt > 0
        upload_video_disabled = videos_cnt > 0 or images_cnt > 0
        _question, _chat_bot, _app_cfg = respond(_question, _chat_bot, _app_cfg)
        return _question, _chat_bot, _app_cfg


def flushed():
    return gr.update(interactive=True)


def clear(txt_message, chat_bot, app_session):
    txt_message.files.clear()
    txt_message.text = ''
    chat_bot = copy.deepcopy(init_conversation)
    app_session['sts'] = None
    app_session['ctx'] = []
    app_session['images_cnt'] = 0
    app_session['videos_cnt'] = 0
    return create_multimodal_input(), chat_bot, app_session
    

def select_chat_type(_tab, _app_cfg):
    _app_cfg["chat_type"] = _tab
    return _app_cfg


init_conversation = [
    [
        None,
        {
            # The first message of bot closes the typewriter.
            "text": "有什么可以帮助你！",
            "flushing": False
        }
    ],
]


css = """
video { height: auto !important; }
.example label { font-size: 16px;}
"""

introduction = """

## Features:
1. Chat with single image
2. Chat with multiple images
3. Chat with video
4. In-context few-shot learning

Click `How to use` tab to see examples.
"""


with gr.Blocks(css=css) as demo:
    app_session = gr.State({'sts':None,'ctx':[], 'images_cnt': 0, 'videos_cnt': 0, 'chat_type': 'Chat'})

    with gr.Tab("图片理解"):
        with gr.Row():
            # 左侧：输入区域
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="filepath", 
                    sources=["upload"], 
                    label="上传图片"
                )              
            
            # 右侧：聊天区域  
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    value=[],
                    height=500,
                    label="对话历史"
                )
                text_input = gr.Textbox(
                    label="输入问题",
                    placeholder="请输入你的问题...",
                    lines=3
                )
                with gr.Row():
                    submit_btn = gr.Button("提交", variant="primary")
                    clear_btn = gr.Button("清除")

        # 绑定事件
        submit_btn.click(
            chat_with_image,
            inputs=[image_input, text_input, chatbot],
            outputs=[chatbot, text_input]
        )
        
        clear_btn.click(
            clear_chat,
            outputs=[chatbot, text_input]
        )

    with gr.Tab("视频推理"):
        with gr.Row():
            # 左侧：视频输入区域
            with gr.Column(scale=1):
                video_input = gr.Video(
                    sources=["upload"],
                    label="上传视频",
                    height=400
                )
           
            # 右侧：分析结果区域
            with gr.Column(scale=2):
                video_chatbot = gr.Chatbot(
                    value=[],
                    height=500,
                    label="视频分析对话"
                )
                video_text_input = gr.Textbox(
                    label="输入问题",
                    placeholder="请输入关于视频的问题...",
                    lines=3
                )
                with gr.Row():
                    video_submit_btn = gr.Button("提交", variant="primary")
                    video_clear_btn = gr.Button("清除")

        def chat_with_video(video, text, history):
            if video is None and not text.strip():
                gr.Warning("请上传视频或输入问题")
                return history, ""
            
            if video is None:
                gr.Warning("请先上传视频")
                return history, text
                
            try:
                # 构建消息
                message = []
                
                # 编码视频
                encoded_video = encode_video(video)
                message.extend(encoded_video)  # 视频帧列表
                
                # 添加文本问题
                if text.strip():
                    message.append(make_text(text))
                else:
                    message.append(make_text("请分析这个视频的内容"))
                
                context = [{'role': 'user', 'content': message}]
                
                # 设置视频推理参数
                params = {
                    'sampling': True,
                    'top_p': 0.8,
                    'top_k': 100,
                    'temperature': 0.7,
                    'repetition_penalty': 1.05,
                    "max_new_tokens": 2048,
                    "max_inp_length": 4352,  # 视频需要更长的输入长度
                    "use_image_id": False,
                    "max_slice_nums": 1 if len(encoded_video) > 16 else 2,  # 根据帧数调整
                    # "enable_thinking": True
                }
                                
                code, answer, _, _ = chat("", context, None, params)
                
                if code == 0:
                    user_message = text if text.strip() else "分析视频内容"
                    history.append((user_message, answer))
                    return history, ""
                else:
                    gr.Warning("视频分析时出错")
                    return history, text
                    
            except Exception as e:
                print(f"视频分析错误: {e}")
                traceback.print_exc()
                gr.Warning(f"处理失败: {str(e)}")
                return history, text

        def clear_video_chat():
            return [], ""

        # 绑定视频推理事件
        video_submit_btn.click(
            chat_with_video,
            inputs=[video_input, video_text_input, video_chatbot],
            outputs=[video_chatbot, video_text_input]
        )
        
        video_text_input.submit(
            chat_with_video,
            inputs=[video_input, video_text_input, video_chatbot],
            outputs=[video_chatbot, video_text_input]
        )
        
        video_clear_btn.click(
            clear_video_chat,
            outputs=[video_chatbot, video_text_input]
        )

        with gr.Row():
            # 左侧空白占位
            with gr.Column(scale=1):
                gr.Markdown("")
            
            with gr.Column(scale=2):
                gr.Examples(
                    examples=[
                        ["描述视频中发生了什么"],
                        ["视频中有哪些物体或人物？"],
                        ["分析视频中的动作或行为"],
                        ["视频的场景是什么？"],
                        ["总结视频的主要内容"]
                    ],
                    inputs=[video_text_input],
                    label="常用视频分析问题"
                )

    with gr.Tab("场景变化检测"):
        with gr.Column():
            with gr.Row():
                image1_input = gr.Image(type="filepath", sources=["upload"], label="第一张图片", height=400)
                image2_input = gr.Image(type="filepath", sources=["upload"], label="第二张图片", height=400)

            with gr.Row():
                scene_change_output = gr.Textbox(
                    label="场景变化分析结果", 
                    lines=10, 
                    interactive=False,
                    placeholder="检测结果将在这里显示..."
                )

            with gr.Row():
                scene_change_button = gr.Button("检测场景变化", variant="primary")

            def detect_scene_change(img1, img2):
                if img1 is None or img2 is None:
                    return "请上传两张图片进行场景变化检测"
                
                try:
                    # 编码两张图片
                    image1 = encode_image(img1)
                    image2 = encode_image(img2)
                    
                    # 构建消息
                    message = [
                        make_text("第一张图片："),
                        image1,
                        make_text("第二张图片："),
                        image2, 
                        make_text("忽略：光照，先直接回答：有无明显变化，再总结变化内容，简短点")
                    ]
                    
                    context = [{'role': 'user', 'content': message}]
                    
                    # 设置参数
                    params = {
                        'sampling': True,
                        'top_p': 0.8,
                        'top_k': 100,
                        'temperature': 0.7,
                        'repetition_penalty': 1.05,
                        "max_new_tokens": 2048,
                        "max_slice_nums": 2
                    }
                    
                    # 调用模型
                    code, answer, _, _ = chat("", context, None, params)
                    
                    if code == 0:
                        return answer
                    else:
                        return "检测过程中发生错误，请重试"
                        
                except Exception as e:
                    print(f"场景变化检测错误: {e}")
                    traceback.print_exc()
                    return f"检测失败: {str(e)}"
            
            scene_change_button.click(
                detect_scene_change,
                inputs=[image1_input, image2_input],
                outputs=[scene_change_output]
            )



# launch
demo.launch(share=True, debug=True, show_api=False, server_port=8001, server_name="0.0.0.0")
