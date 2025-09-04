import streamlit as st
import os
import pandas as pd
from typing import List, Optional

class VideoManager:
    def __init__(self):
        self.videos = []
    
    def add_videos(self, video_files: List) -> str:
        """添加视频到列表"""
        if video_files:
            added_count = 0
            for video_file in video_files:
                if video_file and video_file.name not in [os.path.basename(v) for v in self.videos]:
                    # 保存上传的文件
                    file_path = os.path.join("temp_videos", video_file.name)
                    os.makedirs("temp_videos", exist_ok=True)
                    with open(file_path, "wb") as f:
                        f.write(video_file.read())
                    self.videos.append(file_path)
                    added_count += 1
            return f"已添加 {added_count} 个视频，当前总计 {len(self.videos)} 个视频"
        return "未选择视频文件"
    
    def get_video_dataframe(self) -> pd.DataFrame:
        """获取视频列表的DataFrame"""
        if not self.videos:
            return pd.DataFrame(columns=["索引", "视频文件"])
        
        data = [[i, os.path.basename(video)] for i, video in enumerate(self.videos)]
        return pd.DataFrame(data, columns=["索引", "视频文件"])
    
    def get_video_path(self, index: int) -> Optional[str]:
        """根据索引获取视频路径"""
        if 0 <= index < len(self.videos):
            return self.videos[index]
        return None
    
    def remove_video(self, index: int) -> str:
        """移除指定索引的视频"""
        if 0 <= index < len(self.videos):
            removed_video = os.path.basename(self.videos[index])
            # 删除文件
            if os.path.exists(self.videos[index]):
                os.remove(self.videos[index])
            self.videos.pop(index)
            return f"已移除视频: {removed_video}"
        return "无效的索引"
    
    def clear_videos(self) -> str:
        """清空所有视频"""
        count = len(self.videos)
        # 删除所有临时文件
        for video_path in self.videos:
            if os.path.exists(video_path):
                os.remove(video_path)
        self.videos.clear()
        return f"已清空 {count} 个视频"

# 初始化 session state
if 'video_manager' not in st.session_state:
    st.session_state.video_manager = VideoManager()

if 'selected_video_index' not in st.session_state:
    st.session_state.selected_video_index = None

if 'status_message' not in st.session_state:
    st.session_state.status_message = ""

# 页面标题
st.set_page_config(page_title="视频管理应用", layout="wide")
st.title("视频管理应用")
st.markdown("上传多个视频文件，在列表中查看和管理")

# 创建三列布局
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    st.markdown("### 上传视频")
    
    # 视频上传
    uploaded_files = st.file_uploader(
        "选择视频文件",
        type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
        accept_multiple_files=True,
        key="video_uploader"
    )
    
    if st.button("添加视频", type="primary"):
        if uploaded_files:
            message = st.session_state.video_manager.add_videos(uploaded_files)
            st.session_state.status_message = message
            st.rerun()
    
    st.markdown("### 操作")
    
    # 删除视频
    col1_1, col1_2 = st.columns([2, 1])
    with col1_1:
        remove_index = st.number_input(
            "输入要删除的视频索引",
            min_value=0,
            max_value=len(st.session_state.video_manager.videos) - 1 if st.session_state.video_manager.videos else 0,
            step=1,
            key="remove_index"
        )
    with col1_2:
        if st.button("删除", type="secondary"):
            message = st.session_state.video_manager.remove_video(remove_index)
            st.session_state.status_message = message
            st.session_state.selected_video_index = None
            st.rerun()
    
    # 清空所有视频
    if st.button("清空所有视频", type="secondary"):
        message = st.session_state.video_manager.clear_videos()
        st.session_state.status_message = message
        st.session_state.selected_video_index = None
        st.rerun()
    
    # 状态信息
    st.markdown("### 状态信息")
    if st.session_state.status_message:
        st.info(st.session_state.status_message)

with col2:
    st.markdown("### 视频列表")
    
    # 显示视频列表
    video_df = st.session_state.video_manager.get_video_dataframe()
    
    if not video_df.empty:
        # 使用 dataframe 显示，并添加选择功能
        selected_indices = st.dataframe(
            video_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row"
        )
        
        # 处理选择事件
        if selected_indices.selection.rows:
            selected_row = selected_indices.selection.rows[0]
            st.session_state.selected_video_index = selected_row
    else:
        st.info("暂无视频文件")


with col3:
    st.markdown("### 视频预览")
    
    # 视频预览
    if (st.session_state.selected_video_index is not None and 
        st.session_state.selected_video_index < len(st.session_state.video_manager.videos)):
        
        video_path = st.session_state.video_manager.get_video_path(st.session_state.selected_video_index)
        if video_path and os.path.exists(video_path):
            # 创建一个有固定高度的容器
            width = 
            with st.container(height=400):
                
                # 或者直接使用 streamlit 的视频组件（推荐）
                st.video(video_path, format="video/mp4", start_time=0)
                
                st.caption(f"正在播放: {os.path.basename(video_path)}")
                
                # 添加视频信息
                try:
                    file_size = os.path.getsize(video_path)
                    st.text(f"文件大小: {file_size / (1024*1024):.2f} MB")
                except:
                    pass
        else:
            st.warning("视频文件不存在")
    else:
        st.info("请从左侧列表选择视频进行预览")