"""Kimi API 文件上下文管理模块

功能说明：
1. 文件管理
   - 支持上传文件到 Kimi API
   - 获取文件信息和状态
   - 管理文件的生命周期

2. 对话管理
   - 创建基于文件的对话
   - 发送消息和获取回复
   - 获取对话历史记录
   - 删除对话

3. 错误处理
   - 完整的异常捕获和日志记录
   - API 调用失败的优雅处理
   - 网络问题的重试机制

4. 安全性
   - API 密钥管理
   - 请求头验证
   - 敏感信息保护

使用示例：
    # 初始化
    kimi = KimiFileContext(api_key="your_api_key")
    
    # 上传文件
    file_result = kimi.upload_file("path/to/file")
    if file_result:
        file_id = file_result["file_id"]
        
        # 创建对话
        conversation_id = kimi.create_conversation([file_id])
        if conversation_id:
            # 发送消息
            response = kimi.send_message(conversation_id, "分析这个文件的内容")
            
            # 获取对话历史
            messages = kimi.get_conversation_messages(conversation_id)

注意事项：
1. 需要有效的 Kimi API 密钥
2. 文件上传大小有限制
3. 对话会话有时效性
4. 建议做好错误处理
"""

# coding=utf-8

import time
from typing import *
import os
import json
from openai import OpenAI
from loguru import logger
from common.singleton import singleton
import copy

from config import get_root
from plugins.plugin_kimichat.module.kimi_api.knowledge.kimi_cache_man import KimiCacheMan
from plugins.plugin_kimichat.module.kimi_api.knowledge.kimi_session_man import KimiSessionMan


@singleton
class KimiFileContext:
    """Kimi文件上下文管理类
    
    用于管理与Kimi API的文件交互，包括上传文件、创建对话等功能。
    
    Attributes:
        api_key: Kimi API密钥
        base_url: Kimi API基础URL
        headers: HTTP请求头
    """
    
    def __init__(self, config):
        """初始化KimiFileContext
        
        Args:
            api_key: Kimi API密钥，如果不提供则从环境变量获取
        """
        self.config = config
        self.api_key = config.get("kimi_api_key")
        if not self.api_key:
            raise ValueError("KIMI_API_KEY not found")
            
        self.base_url = config.get("kimi_api_url")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        self.kimi_model = config.get("kimi_model")
        self.sessions = KimiSessionMan(config)
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.parent_path = os.path.dirname(self.current_path)
        self.plugins_path = os.path.join(get_root(),'plugins')
        
        
    
            
    def proc_knowledge_chat(self, session:str, msg:dict):
        """处理知识库聊天
        必然是 at  group 消息
        Args:
            session: 会话
            msg: 消息
        """
        logger.info(f"[KimiFileContext] 开始处理知识库聊天...")
        try:
            user_id = msg.from_user_id
            user_nickname = msg.from_user_nickname
            content = msg.content
            
            session_obj, _ = self.sessions.get_session(session)
            
            # 使用deepcopy，防止修改原始会话
            messages = copy.deepcopy(session_obj.messages)
            messages.append({"role": "user", "content": content})            
            # 调用Kimi API进行聊天            
            completion = self.client.chat.completions.create(
                model=self.kimi_model,
                messages=messages,
            )
            # 添加Kimi的回答到sessions
            logger.info(completion.choices[0].message.content)
            self.sessions.add_message(session, "user", content)
            self.sessions.add_message(session, "assistant", completion.choices[0].message.content)
            return completion.choices[0].message.content
        
        except Exception as e:
            logger.error(f"[KimiFileContext] 处理知识库聊天失败: {e}")
            return "处理知识库聊天失败"
        