import torch
import gradio as gr
import os
import time
import json
import shutil
from typing import List, Optional
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoModel, AutoTokenizer


# 常量定义
MAX_NUM_FRAMES = 64
FRAME_GAP = 1
CACHE_DIR = "video_cache"
JSON_FILE = "video_list.json"
DEFAULT_CONVERSATION = ""


video_inp_params = {
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

model_path = '/home/rhs/code_workspace/VLM/hf_models/MiniCPM-V-4_5'
model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) 
model = model.eval().to('cuda:0')
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)

# 创建缓存目录
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


def encode_image(image):
    """编码和调整图像大小。
    
    Args:
        image: 输入图像，可以是PIL Image对象或文件路径
        
    Returns:
        PIL.Image: 处理后的图像对象
    """
    if not isinstance(image, Image.Image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif hasattr(image, 'path'):
            image = Image.open(image.path).convert("RGB")
        else:
            image = Image.open(image.file.path).convert("RGB")
    
    # 调整图像大小
    max_size = 448 * 16 
    if max(image.size) > max_size:
        w, h = image.size
        if w > h:
            new_w = max_size
            new_h = int(h * max_size / w)
        else:
            new_h = max_size
            new_w = int(w * max_size / h)
        image = image.resize((new_w, new_h), resample=Image.BICUBIC)
    return image


def encode_video(video):
    """编码视频，提取关键帧。
    
    Args:
        video: 视频文件路径或对象
        
    Returns:
        List[PIL.Image]: 提取的视频帧列表
    """
    def uniform_sample(l, n):
        """均匀采样函数"""
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    # 获取视频路径
    if isinstance(video, str):
        video_path = video
    elif hasattr(video, 'path'):
        video_path = video.path
    elif hasattr(video, 'file') and hasattr(video.file, 'path'):
        video_path = video.file.path
    else:
        video_path = str(video)
    
    vr = VideoReader(video_path, ctx=cpu(0))

    sample_fps = round(vr.get_avg_fps() / FRAME_GAP)
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    
    video_frames = vr.get_batch(frame_idx).asnumpy()
    video_frames = [Image.fromarray(v.astype('uint8')) for v in video_frames]
    video_frames = [encode_image(v) for v in video_frames]
    print(f'视频帧数: {len(video_frames)}')
    return video_frames


class VideoManager:
    """视频管理器类，负责视频的添加、删除、缓存管理和自动保存加载"""
    
    def __init__(self):
        """初始化视频管理器"""
        self.video_json = {}
        self.video_id_counter = 0
        self._load_json_automatically()
    
    def _save_frames_to_cache(self, frames: List[Image.Image], video_id: int) -> List[str]:
        """将帧图像保存到本地缓存并返回路径列表"""
        video_cache_dir = os.path.join(CACHE_DIR, f"video_{video_id}")
        if not os.path.exists(video_cache_dir):
            os.makedirs(video_cache_dir)
        
        frame_paths = []
        for i, frame in enumerate(frames):
            frame_filename = f"frame_{i:04d}.jpg"
            frame_path = os.path.join(video_cache_dir, frame_filename)
            frame.save(frame_path, "JPEG", quality=95)
            frame_paths.append(frame_path)
        
        return frame_paths
    
    def _load_frames_from_cache(self, frame_paths: List[str]) -> List[Image.Image]:
        """从缓存路径加载帧图像"""
        frames = []
        for path in frame_paths:
            if os.path.exists(path):
                loaded_image = Image.open(path)
                frames.append(loaded_image)
        return frames
    
    def _save_json_automatically(self):
        """自动保存JSON到文件"""
        try:
            save_data = {}
            for vid_id, vid_data in self.video_json.items():
                save_data[vid_id] = {
                    "video_name": vid_data["video_name"],
                    "timestamp": vid_data["timestamp"],
                    "frames": vid_data["frames"],
                    "video_id": vid_data["video_id"],
                    "frame_count": vid_data["frame_count"],
                    "status": vid_data["status"]
                }
            
            with open(JSON_FILE, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"自动保存JSON失败: {e}")
    
    def _load_json_automatically(self):
        """自动从文件加载JSON"""
        if not os.path.exists(JSON_FILE):
            return
        
        try:
            with open(JSON_FILE, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            
            for vid_id, vid_data in loaded_data.items():
                # 检查视频文件是否存在
                if os.path.exists(vid_data["video_name"]):
                    # 检查帧文件是否存在
                    frames_exist = True
                    if "frames" in vid_data:
                        for frame_path in vid_data["frames"]:
                            if not os.path.exists(frame_path):
                                frames_exist = False
                                break
                    
                    if frames_exist and "frames" in vid_data:
                        # 直接加载现有的帧路径
                        self.video_json[int(vid_id)] = vid_data
                    else:
                        # 重新处理视频以获取frames
                        try:
                            images = encode_video(vid_data["video_name"])
                            frame_paths = self._save_frames_to_cache(images, int(vid_id))
                            self.video_json[int(vid_id)] = {
                                **vid_data,
                                "frames": frame_paths,
                                "frame_count": len(frame_paths)
                            }
                        except Exception as e:
                            print(f"重新处理视频 {vid_data['video_name']} 时出错: {e}")
            
            if self.video_json:
                self.video_id_counter = max(self.video_json.keys()) + 1
                
        except Exception as e:
            print(f"自动加载JSON失败: {e}")
    
    def add_videos(self, video_files: List[str]) -> str:
        """添加视频到管理器"""
        if not video_files:
            return "未选择视频文件"
        
        added_count = 0
        for video_name in video_files:
            if not video_name:
                continue
                
            # 检查是否已存在
            existing = any(
                vid_data.get("video_name") == video_name 
                for vid_data in self.video_json.values()
            )
            
            if not existing:
                try:
                    images = encode_video(video=video_name)
                    frame_paths = self._save_frames_to_cache(images, self.video_id_counter)
                    
                    self.video_json[self.video_id_counter] = {
                        "video_name": video_name,
                        "timestamp": time.time(),
                        "frames": frame_paths,
                        "video_id": self.video_id_counter,
                        "frame_count": len(frame_paths),
                        "status": "processed"
                    }
                    self.video_id_counter += 1
                    added_count += 1
                except Exception as e:
                    print(f"处理视频 {video_name} 时出错: {e}")
                    continue
        
        # 自动保存
        self._save_json_automatically()
        return f"已添加 {added_count} 个视频，当前总计 {len(self.video_json)} 个视频"
    
    def get_video_list(self) -> List[List[str]]:
        """获取视频列表，返回二维数组格式供Dataframe使用"""
        video_list = []
        for vid_id, vid_data in self.video_json.items():
            video_list.append([
                str(vid_id),
                os.path.basename(vid_data["video_name"]),
                str(vid_data["frame_count"]),
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(vid_data["timestamp"])),
                DEFAULT_CONVERSATION  # 添加空的对话列
            ])
        return video_list
    
    def get_video_path(self, video_id: int) -> Optional[str]:
        """根据视频ID获取视频路径"""
        return self.video_json.get(video_id, {}).get("video_name")
    
    def get_video_frames(self, video_id: int) -> Optional[List[str]]:
        """根据视频ID获取视频帧路径列表"""
        return self.video_json.get(video_id, {}).get("frames")
    
    def get_video_frame_images(self, video_id: int) -> Optional[List[Image.Image]]:
        """根据视频ID获取视频帧图像对象列表"""
        frame_paths = self.get_video_frames(video_id)
        if frame_paths:
            try:
                return self._load_frames_from_cache(frame_paths)
            except Exception as e:
                print(f"加载视频帧图像时出错: {e}")
                return None
        return None
    
    def remove_video(self, video_id: int) -> str:
        """移除指定ID的视频"""
        if video_id not in self.video_json:
            return "无效的视频ID"
        
        removed_video = os.path.basename(self.video_json[video_id]["video_name"])
        
        # 删除缓存的帧文件
        frame_paths = self.video_json[video_id]["frames"]
        video_cache_dir = os.path.join(CACHE_DIR, f"video_{video_id}")
        try:
            for frame_path in frame_paths:
                if os.path.exists(frame_path):
                    os.remove(frame_path)
            if os.path.exists(video_cache_dir):
                os.rmdir(video_cache_dir)
        except Exception as e:
            print(f"删除缓存文件时出错: {e}")
        
        del self.video_json[video_id]
        
        # 自动保存
        self._save_json_automatically()
        return f"已移除视频: {removed_video}"
    
    def clear_all_cache(self) -> str:
        """清除所有本地缓存和数据"""
        count = len(self.video_json)
        
        # 删除所有缓存文件
        try:
            if os.path.exists(CACHE_DIR):
                shutil.rmtree(CACHE_DIR)
                os.makedirs(CACHE_DIR)  # 重新创建空目录
        except Exception as e:
            print(f"删除缓存目录时出错: {e}")
        
        # 删除JSON文件
        try:
            if os.path.exists(JSON_FILE):
                os.remove(JSON_FILE)
        except Exception as e:
            print(f"删除JSON文件时出错: {e}")
        
        # 清空内存数据
        self.video_json.clear()
        self.video_id_counter = 0
        
        return f"已清除所有本地缓存：{count} 个视频记录和所有图片缓存"


# 创建全局VideoManager实例
video_manager = VideoManager()


def add_videos_handler(video_files):
    """处理视频上传"""
    if video_files is None:
        return "未选择视频文件", video_manager.get_video_list(), None
    
    # 提取文件路径
    if isinstance(video_files, list):
        file_paths = [f.name if hasattr(f, 'name') else str(f) for f in video_files]
    else:
        file_paths = [video_files.name if hasattr(video_files, 'name') else str(video_files)]
    
    message = video_manager.add_videos(file_paths)
    updated_list = video_manager.get_video_list()
    return message, updated_list, None


def select_video_handler(evt: gr.SelectData):
    """处理视频选择"""
    try:
        video_id = int(evt.value)
        video_path = video_manager.get_video_path(video_id)
        return video_path, video_id
    except (ValueError, IndexError, TypeError):
        return None, None


def remove_video_handler(remove_id):
    """处理视频移除"""
    try:
        video_id = int(remove_id)
        message = video_manager.remove_video(video_id)
        updated_list = video_manager.get_video_list()
        return message, updated_list, None, ""
    except (ValueError, TypeError):
        return "请输入有效的视频ID", video_manager.get_video_list(), None, ""


def clear_cache_handler():
    """处理清除缓存"""
    message = video_manager.clear_all_cache()
    return message, [], None

def send_message_handler(message, chat_history_state, selected_video_id_value, video_selection_mode):
    """处理发送消息的功能"""
    if not message.strip():
        return "", chat_history_state, "请输入有效的消息内容"
    
    try:
        video_frames = []
        video_info = ""
        
        if video_selection_mode == "single":
            if selected_video_id_value is None:
                return "", chat_history_state, "请先选择一个视频"
            
            frames = video_manager.get_video_frame_images(selected_video_id_value)
            if not frames:
                return "", chat_history_state, "无法获取视频帧，请重新选择视频"
            
            video_frames = frames
            video_path = video_manager.get_video_path(selected_video_id_value)
            video_info = f"当前分析视频: {os.path.basename(video_path)}"
            
        elif video_selection_mode == "all":
            if not video_manager.video_json:
                return "", chat_history_state, "当前没有可用的视频"
            
            # 按时间戳排序视频数据
            sorted_videos = sorted(
                video_manager.video_json.items(),
                key=lambda x: x[1]["timestamp"]
            )
            
            video_info_list = []
            MAX_FRAMES_PER_VIDEO = 16  # 限制每个视频的最大帧数
            
            # 构建包含时间戳和视频名信息的query
            query_content = [message]
            
            for vid_id, vid_data in sorted_videos:
                frames = video_manager.get_video_frame_images(vid_id)
                if frames:
                    # 获取视频信息
                    video_name = os.path.basename(vid_data["video_name"])
                    timestamp_str = time.strftime(
                        "%Y-%m-%d %H:%M:%S", 
                        time.localtime(vid_data["timestamp"])
                    )
                    
                    # 如果帧数过多，进行采样
                    if len(frames) > MAX_FRAMES_PER_VIDEO:
                        step = len(frames) // MAX_FRAMES_PER_VIDEO
                        frames = frames[::step][:MAX_FRAMES_PER_VIDEO]
                    
                    # 添加视频信息文本
                    video_info_text = f"视频名称: {video_name}, 添加时间: {timestamp_str}, 帧数: {len(frames)}"
                    query_content.append(video_info_text)
                    
                    # 添加该视频的所有帧
                    query_content.extend(frames)
                    
                    video_info_list.append(f"{video_name}({timestamp_str})")
            
            if len(query_content) == 1:  # 只有原始消息，没有添加任何视频内容
                return "", chat_history_state, "无法获取任何视频帧"
            
            video_frames = query_content
            video_info = f"按时间顺序分析 {len(video_info_list)} 个视频: {', '.join(video_info_list)}"
        
        # 构建上下文
        if video_selection_mode == "single":
            query = [message]
            query.extend(video_frames)
        else:  # video_selection_mode == "all"
            query = video_frames  # 已经包含了完整的内容
        
        context = [{
            'role': 'user', 
            'content': query
        }]
        
        print(f"处理查询: {video_info}")
        answer = model.chat(
            msgs=context,
            tokenizer=tokenizer,
            **video_inp_params
        )
        
        new_history = chat_history_state + [[message, answer]]
        return "", new_history, f"已处理消息 ({video_info})，当前对话轮数：{len(new_history)}"
        
    except Exception as e:
        return "", chat_history_state, f"处理消息时出错：{str(e)}"


def clear_chat_handler():
    """清空对话历史"""
    return [], "对话历史已清空"


def update_conversation_placeholder(selected_video_id_value, video_selection_mode):
    """更新对话输入框的占位符"""
    if video_selection_mode == "single":
        if selected_video_id_value is not None:
            video_path = video_manager.get_video_path(selected_video_id_value)
            video_name = os.path.basename(video_path) if video_path else "未知视频"
            return gr.update(
                placeholder=f"正在与视频 '{video_name}' (ID: {selected_video_id_value}) 对话，请输入你的问题...",
                interactive=True
            )
        else:
            return gr.update(
                placeholder="请先选择视频，然后在此输入对话内容...",
                interactive=False
            )
    elif video_selection_mode == "all":
        video_count = len(video_manager.video_json)
        if video_count > 0:
            return gr.update(
                placeholder=f"正在与所有视频 ({video_count} 个) 对话，请输入你的问题...",
                interactive=True
            )
        else:
            return gr.update(
                placeholder="当前没有可用的视频，请先上传视频...",
                interactive=False
            )


def video_selection_mode_change_handler(video_selection_mode, selected_video_id_value):
    """处理视频选择模式变化"""
    return update_conversation_placeholder(selected_video_id_value, video_selection_mode)


# 创建Gradio界面
with gr.Blocks(title="视频管理应用") as demo:
    gr.Markdown("# 视频管理应用")
    gr.Markdown("上传多个视频文件，在列表中查看和管理（自动保存/加载）")
    
    with gr.Row():
        with gr.Column(scale=1):
            # 视频上传区域
            gr.Markdown("### 上传视频")
            video_upload = gr.File(
                label="选择视频文件",
                file_count="multiple",
                file_types=["video"]
            )
            
            upload_btn = gr.Button("添加视频", variant="primary")
            
            # 操作按钮
            gr.Markdown("### 操作")
            with gr.Row():
                remove_id_input = gr.Number(
                    label="输入要删除的视频ID",
                    precision=0,
                    minimum=0
                )
                remove_btn = gr.Button("删除视频", variant="secondary")
            
           
            # 缓存管理
            gr.Markdown("### 缓存管理")
            clear_cache_btn = gr.Button("清除本地缓存", variant="stop")
            gr.Markdown("⚠️ 此操作将删除所有视频记录和图片缓存文件")
            
            # 状态信息
            status_text = gr.Textbox(
                label="状态信息",
                interactive=False,
                lines=3
            )
        
        with gr.Column(scale=2):
            # 视频列表显示
            gr.Markdown("### 视频列表")
            video_list = gr.Dataframe(
                headers=["视频ID", "视频文件", "帧数", "添加时间", "对话内容"],
                datatype=["str", "str", "str", "str", "str"],
                interactive=False,
                label="点击选择视频进行预览和编辑对话",
                value=video_manager.get_video_list()  # 启动时显示已有数据
            )

            # 对话管理
            gr.Markdown("### 对话管理")
            
            # 视频选择模式
            video_selection_mode = gr.Radio(
                choices=[("选择单个视频", "single"), ("选择所有视频", "all")],
                value="single",
                label="视频选择模式",
            )

            # 对话历史显示
            chatbot = gr.Chatbot(
                label="对话历史",
                height=300,
                show_label=True,
                container=True,
                bubble_full_width=False
            )
            
            # 状态变量定义
            selected_video_id = gr.State(value=None)
            chat_history = gr.State(value=[])

            conversation_input = gr.Textbox(
                label="对话内容",
                placeholder="请先选择视频，然后在此输入对话内容...",
                lines=2,
                interactive=True
            )
            
            # 聊天机器人界面
            with gr.Row():
                send_btn = gr.Button("发送消息", variant="primary")
                clear_chat_btn = gr.Button("清空对话", variant="secondary")
            

            
        with gr.Column(scale=1):
            # 视频预览区域
            gr.Markdown("### 视频预览")
            video_preview = gr.Video(
                label="选中的视频",
                interactive=False,
                height=700
            )
    
    # 事件绑定
    upload_btn.click(
        fn=add_videos_handler,
        inputs=[video_upload],
        outputs=[status_text, video_list, video_preview]
    )
    
    video_list.select(
        fn=select_video_handler,
        outputs=[video_preview, selected_video_id]
    ).then(
        fn=update_conversation_placeholder,
        inputs=[selected_video_id, video_selection_mode],
        outputs=[conversation_input]
    )
    
    # 视频选择模式变化事件
    video_selection_mode.change(
        fn=video_selection_mode_change_handler,
        inputs=[video_selection_mode, selected_video_id],
        outputs=[conversation_input]
    )
    
    remove_btn.click(
        fn=remove_video_handler,
        inputs=[remove_id_input],
        outputs=[status_text, video_list, video_preview, remove_id_input]
    )
    
    clear_cache_btn.click(
        fn=clear_cache_handler,
        inputs=[],
        outputs=[status_text, video_list, video_preview]
    )

    # 新增的聊天功能事件绑定
    send_btn.click(
        fn=send_message_handler,
        inputs=[conversation_input, chat_history, selected_video_id, video_selection_mode],
        outputs=[conversation_input, chatbot, status_text]
    ).then(
        fn=lambda x: x,  # 同步聊天历史
        inputs=[chatbot],
        outputs=[chat_history]
    )
    
    # 回车键发送消息
    conversation_input.submit(
        fn=send_message_handler,
        inputs=[conversation_input, chat_history, selected_video_id, video_selection_mode],
        outputs=[conversation_input, chatbot, status_text]
    ).then(
        fn=lambda x: x,
        inputs=[chatbot],
        outputs=[chat_history]
    )
    
    # 清空对话
    clear_chat_btn.click(
        fn=clear_chat_handler,
        inputs=[],
        outputs=[chatbot, status_text]
    ).then(
        fn=lambda: [],
        inputs=[],
        outputs=[chat_history]
    )