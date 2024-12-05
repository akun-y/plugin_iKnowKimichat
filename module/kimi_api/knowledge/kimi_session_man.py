"""会话管理类, 管理每个用户的会话状态

主要功能:
1. 每个用户一个会话, 会话中包含多个消息历史记录
2. 会话中有基于文件hash的缓存系统, 用于会话知识库管理
3. 会话知识库支持动态扩展, 可根据聊天内容增加临时知识库
4. 会话状态持久化到本地文件系统, 支持程序重启后恢复
5. 支持会话级别的上下文数据存储和管理
6. 基于缓存命中率的自适应缓存策略, 高命中率时自动延长缓存有效期

技术特点:
1. 使用SQLite实现文件缓存, 支持多线程访问
2. 使用JSON格式持久化会话数据
3. 单例模式确保全局唯一实例
4. 支持会话过期自动清理
5. 线程安全的缓存和会话管理
"""
import os
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

import httpx
from loguru import logger
from openai import OpenAI
from common.singleton import singleton
from config import conf, get_root
from .kimi_cache_man import CacheMessage, KimiCacheMan

@dataclass
class Session:
    """单个用户会话"""
    user_id: str
    messages: List[Dict[str, Any]]
    cache_hit_count: int = 0
    cache_miss_count: int = 0
    last_active_time: float = None
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.last_active_time is None:
            self.last_active_time = time.time()
        if self.context is None:
            self.context = {}
            
    def add_message(self, role: str, content: str):
        """添加消息"""
        self.messages.append({
            'role': role,
            'content': content,
            'timestamp': time.time()
        })
        self.last_active_time = time.time()
        
    def get_cache_hit_rate(self) -> float:
        """获取缓存命中率"""
        total = self.cache_hit_count + self.cache_miss_count
        return self.cache_hit_count / total if total > 0 else 0
        
    def to_dict(self) -> dict:
        """转换为字典用于序列化"""
        return {
            'user_id': self.user_id,
            'messages': self.messages,
            'cache_hit_count': self.cache_hit_count,
            'cache_miss_count': self.cache_miss_count,
            'last_active_time': self.last_active_time,
            'context': self.context
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Session':
        """从字典反序列化"""
        return cls(
            user_id=data['user_id'],
            messages=data['messages'],
            cache_hit_count=data['cache_hit_count'],
            cache_miss_count=data['cache_miss_count'],
            last_active_time=data['last_active_time'],
            context=data['context']
        )

@singleton
class KimiSessionMan:
    """Kimi会话管理类"""
    
    def __init__(self, config, save_path: str = "sessions_files"):
        """初始化会话管理器
        
        Args:
            save_path: 会话保存路径
        """
        root_path = get_root()
        knowledge_file_path = os.path.join(root_path,"plugins","plugin_kimichat","knowledge_files")
        knowledge_file_name = config.get("common_knowledge_file_name","common_knowledge.txt")
        self.knowledge_file = os.path.join(knowledge_file_path, knowledge_file_name)
        

        
        self.sessions: Dict[str, Session] = {}
        self.cache_man = KimiCacheMan(config)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(current_dir, save_path)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 加载会话后立即清理过期会话
        self._load_sessions()
        self._last_cleanup = time.time()
        expired, cleaned = self.clear_expired_sessions()
        if expired or cleaned:
            logger.info(f"[KimiSessionMan] 初始化时清理: {expired}个过期会话, {cleaned}条过长消息")
    
    
    def get_session(self, user_id: str) -> Tuple[Session, bool]:
        """获取用户会话，不存在则创建新会话
        
        Returns:
            Tuple[Session, bool]: (会话对象, 是否为新创建的会话)
        """
        # 每隔一定时间（比如1小时）检查是否需要清理
        current_time = time.time()
        if current_time - self._last_cleanup > 3600:  # 1小时
            expired, cleaned = self.clear_expired_sessions()
            if expired or cleaned:
                logger.info(f"[KimiSessionMan] 定期清理: {expired}个过期会话, {cleaned}条过长消息")
            self._last_cleanup = current_time
        
        is_new = False
        if user_id not in self.sessions:
            # 获取缓存消息并转换为 Message 列表
            cache_messages = self.cache_man.get_cache(self.knowledge_file)
            initial_messages = [{'role':'system','content':conf().get("character_desc")}]
            if cache_messages:
                # 假设 cache_messages 包含 role 和 content 字段
                for msg in cache_messages:
                    initial_messages.append({'role':msg.role,'content':msg.content})
            
            self.sessions[user_id] = Session(
                user_id=user_id,
                messages=initial_messages,
                context={}
            )   
            is_new = True
            logger.warning(f"[KimiSessionMan] 创建新会话: {user_id}")
        else:
            logger.info(f"[KimiSessionMan] 获取原有会话: {user_id},会话长度: {len(self.sessions[user_id].messages)}")
        return self.sessions[user_id], is_new
        
    def add_message(self, user_id: str, role: str, content: str):
        """添加消息到用户会话"""
        session, _ = self.get_session(user_id)
        session.add_message(role, content)
        self._save_session(session)
        
    def add_temp_knowledge(self, user_id: str, file_path: str, 
                          expire_seconds: Optional[int] = 3600) -> Optional[str]:
        """添加临时知识库
        
        Returns:
            Optional[str]: 文件缓存ID，如果添加失败返回None
        """
        session, _ = self.get_session(user_id)
        cache = self.cache_man.get_cache(file_path)
        
        if cache:
            session.cache_hit_count += 1
            # 根据命中率动态调整缓存时间
            hit_rate = session.get_cache_hit_rate()
            if hit_rate > 0.8:  # 命中率高时延长缓存时间
                self.cache_man.refresh_cache(file_path, expire_seconds * 2)
            return cache.get('file_id')
        else:
            session.cache_miss_count += 1
            # 添加新文件到缓存
            self.cache_man.set_cache(
                file_path=file_path,
                file_id=None,  # 由调用方设置file_id
                message="Added from session",
                cache_tag=user_id,
                expire_seconds=expire_seconds
            )
            return None
            
    def set_context(self, user_id: str, key: str, value: Any):
        """设置会话上下文"""
        session, _ = self.get_session(user_id)
        session.context[key] = value
        self._save_session(session)
        
    def get_context(self, user_id: str, key: str) -> Optional[Any]:
        """获取会话上下文"""
        session, _ = self.get_session(user_id)
        return session.context.get(key)
        
    def _save_session(self, session: Session):
        """保存单个会话到文件"""
        file_path = os.path.join(self.save_dir, f"{session.user_id}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
            
    def _load_sessions(self):
        """从文件加载所有会话"""
        if not os.path.exists(self.save_dir):
            return
            
        for filename in os.listdir(self.save_dir):
            if not filename.endswith('.json'):
                continue
                
            file_path = os.path.join(self.save_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    session = Session.from_dict(data)
                    self.sessions[session.user_id] = session
            except Exception as e:
                print(f"Failed to load session {filename}: {e}")
                
    def _deduplicate_messages(self, messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        """去除重复消息，保留最后一条
        
        Args:
            messages: 消息列表
            
        Returns:
            Tuple[List[Dict[str, Any]], int]: (去重后的消息列表, 去除的重复消息数)
        """
        # 分离system消息和其他消息
        system_messages = [msg for msg in messages if msg['role'] == 'system']
        other_messages = [msg for msg in messages if msg['role'] != 'system']
        
        # 对非系统消息进行去重，保留最后一条
        seen_contents = {}
        unique_messages = []
        for msg in reversed(other_messages):  # 从后向前遍历，保留最新的
            content = msg['content']
            if content not in seen_contents:
                seen_contents[content] = True
                unique_messages.append(msg)
        
        # 反转回原来的顺序
        unique_messages.reverse()
        
        # 计算去除的重复消息数
        duplicates_removed = len(other_messages) - len(unique_messages)
        
        # 按照原有顺序重组消息列表：system消息在前，其他消息按时间顺序
        logger.warning(f"[KimiSessionMan] 去重消息: {duplicates_removed} 条")
        return system_messages + unique_messages, duplicates_removed

    def clear_expired_sessions(self, expire_seconds: int = 3*24*3600, max_messages: int = 300):
        """清理过期会话和过长的消息队列
        
        清理规则:
        1. 清理超过指定时间未活动的会话
        2. 对所有会话执行消息去重
        3. 对于消息队列超过max_messages的会话:
           - 保留所有system角色的消息
           - 对其他消息执行先进先出的清理，直到队列长度符合要求
        
        Args:
            expire_seconds: 过期时间(秒)，默认3天
            max_messages: 最大消息数量，默认300条
        
        Returns:
            Tuple[int, int]: (清理的会话数, 清理的消息数)
        """
        current_time = time.time()
        expired_users = []
        total_cleaned_messages = 0
        
        # 遍历所有会话
        for user_id, session in self.sessions.items():
            # 检查会话是否过期
            if current_time - session.last_active_time > expire_seconds:
                expired_users.append(user_id)
                continue
            
            messages = session.messages
            original_count = len(messages)
            
            # 先执行去重操作
            deduped_messages, duplicates_removed = self._deduplicate_messages(messages)
            total_cleaned_messages += duplicates_removed
            
            # 如果去重后仍超过最大限制，进行截断
            if len(deduped_messages) > max_messages:
                # 分离system消息和其他消息
                system_messages = [msg for msg in deduped_messages if msg['role'] == 'system']
                other_messages = [msg for msg in deduped_messages if msg['role'] != 'system']
                
                # 计算需要保留的其他消息数量
                keep_count = max_messages - len(system_messages)
                if keep_count > 0:
                    kept_messages = other_messages[-keep_count:]
                else:
                    kept_messages = []
                
                # 更新会话消息
                truncated_count = len(deduped_messages) - (len(system_messages) + len(kept_messages))
                total_cleaned_messages += truncated_count
                
                # 重组消息列表
                session.messages = system_messages + kept_messages
            else:
                session.messages = deduped_messages
            
            # 如果有任何更改，记录日志并保存
            if len(session.messages) != original_count:
                logger.info(f"[KimiSessionMan] 清理会话 {user_id} 的消息队列: "
                          f"原始消息数: {original_count}, "
                          f"去重后消息数: {len(deduped_messages)}, "
                          f"最终保留消息数: {len(session.messages)} "
                          f"(系统消息: {sum(1 for msg in session.messages if msg['role'] == 'system')})")
                self._save_session(session)
        
        # 删除过期会话
        for user_id in expired_users:
            logger.warning(f"[KimiSessionMan] 删除过期会话: {user_id}")
            del self.sessions[user_id]
            file_path = os.path.join(self.save_dir, f"{user_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
        
        if expired_users or total_cleaned_messages:
            logger.info(f"[KimiSessionMan] 清理完成: "
                      f"删除 {len(expired_users)} 个过期会话, "
                      f"清理 {total_cleaned_messages} 条重复或过长消息")
        
        return len(expired_users), total_cleaned_messages
