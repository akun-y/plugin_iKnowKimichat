"""缓存管理类,专用于kimi_chat,减少token消耗,文件在网络上的传输

主要功能:
1. 缓存记录保存到sqlite中,确保可多线程访问 
2. 为每个文件建立以文件hash为key的缓存记录
3. 可以为一组文件定义 cache_tag, 并定义缓存有效期
4. 缓存有效期到了后, 自动删除缓存
5. 缓存可以设置为自动续期, 即在缓存有效期内访问同一个文件时自动续期
6. 缓存记录中保存文件的hash、路径、消息历史、标签、过期时间等信息
7. 缓存未命中时, 根据缓存记录中的文件路径读取文件内容并更新缓存记录

技术特点:
1. 使用SQLite实现持久化存储
2. 线程安全的缓存访问
3. 基于文件hash的缓存标识
4. 支持缓存分组管理
5. 自动过期清理机制
"""

import sqlite3
import hashlib
import time
import threading
import os
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from loguru import logger
from openai import OpenAI

from plugins.plugin_comm.dot_dict import DotDict
from plugins.plugin_kimichat.module.kimi_api.knowledge import kimi_upload_file

@dataclass
class CacheMessage:
    """缓存消息"""
    role: str
    content: str
    expired_time: Optional[int] = None  # 消息过期时间戳
    
    def is_expired(self) -> bool:
        """检查消息是否过期"""
        if self.expired_time is None:
            return False
        return int(time.time()) > self.expired_time

class KimiCacheMan:
    """Kimi聊天文件缓存管理类"""
    
    def __init__(self,config, db_name: str = "kimi_cache.db"):
        """初始化缓存管理器
        
        Args:
            db_name: SQLite数据库文件名
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(current_dir, db_name)
        self.lock = threading.Lock()
        self._init_db()
        
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
    
    def _init_db(self):
        """初始化数据库表结构"""
        with self._get_conn() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS file_cache (
                    file_hash TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    messages TEXT,  -- JSON格式存储消息数组
                    cache_tag TEXT,
                    cache_expired_time INTEGER,
                    cache_message TEXT,
                    create_time INTEGER,
                    update_time INTEGER
                )
            ''')
    
    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接"""
        return sqlite3.connect(self.db_path)
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件的MD5哈希值
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件的MD5哈希值
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _serialize_messages(self, messages: List[CacheMessage]) -> str:
        """序列化消息列表为JSON字符串"""
        return json.dumps([{
            'role': msg.role, 
            'content': msg.content,
            'expired_time': msg.expired_time
        } for msg in messages])
    
    def _deserialize_messages(self, messages_json: str) -> List[CacheMessage]:
        """从JSON字符串反序列化消息列表，并过滤掉过期消息"""
        if not messages_json:
            return []
        messages_data = json.loads(messages_json)
        messages = []
        for msg in messages_data:
            cache_msg = CacheMessage(
                role=msg['role'],
                content=msg['content'],
                expired_time=msg.get('expired_time')
            )
            if not cache_msg.is_expired():
                messages.append(cache_msg)
        return messages
    
    def get_cache(self, file_path: str) -> Optional[List[CacheMessage]]:
        """获取文件的缓存记录中的消息列表
        
        Args:
            file_path: 文件路径
            
        Returns:
            消息列表，未命中或缓存过期返回None。返回的消息已过滤掉过期消息。
        """
        if not os.path.exists(file_path):
            return None
            
        file_hash = self._calculate_file_hash(file_path)
        
        with self.lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    'SELECT messages, cache_expired_time FROM file_cache WHERE file_hash = ?',
                    (file_hash,)
                )
                row = cursor.fetchone()
                
                if row:
                    # 检查缓存是否过期
                    current_time = int(time.time())
                    if row[1] and current_time > row[1]:  # cache_expired_time
                        logger.warning(f"[KimiCacheMan] 缓存过期, 自动续期: {file_path}")
                        
                        
                    return self._deserialize_messages(row[0])

                messages = kimi_upload_file.upload_files(self.client,[file_path])
                if not messages:
                    logger.error("[KimiFileContext] 上传知识库文件失败")
                    return "上传知识库文件失败"
                msgs = [DotDict(msg) for msg in messages]
                self.insert_cache(conn,file_hash,file_path,self._serialize_messages(msgs),
                                 expired_time=None,cache_message=None)
                return msgs   
    
    
    def set_cache(self, file_path: str, messages: List[CacheMessage],
                  cache_tag: Optional[str] = None, 
                  expire_seconds: Optional[int] = None,
                  cache_message: Optional[str] = None) -> None:
        """设置文件缓存
        
        Args:
            file_path: 文件路径
            messages: 消息列表
            cache_tag: 缓存标签
            expire_seconds: 过期时间(秒)
            cache_message: 缓存相关消息
        """
        file_hash = self._calculate_file_hash(file_path)
        current_time = int(time.time())
        expired_time = current_time + expire_seconds if expire_seconds else None
        messages_json = self._serialize_messages(messages)
        
        with self.lock:
            with self._get_conn() as conn:
                self.insert_cache(conn,file_hash,file_path,messages_json,
                                 cache_tag, expired_time, cache_message)
    def insert_cache(self,conn: sqlite3.Connection, file_hash: str, file_path: str, messages_json: str, 
                     cache_tag: Optional[str] = None, 
                     expired_time: Optional[int] = None,
                     cache_message: Optional[str] = None) -> None:
        """插入缓存记录"""
        current_time = int(time.time())
        conn.execute('''
                    INSERT OR REPLACE INTO file_cache
                    (file_hash, file_path, messages, cache_tag,
                     cache_expired_time, cache_message, create_time, update_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (file_hash, file_path, messages_json, cache_tag,
                      expired_time, cache_message, current_time, current_time))
    def remove_cache(self, file_hash: str) -> None:
        """删除缓存记录
        
        Args:
            file_hash: 文件哈希值
        """
        with self.lock:
            with self._get_conn() as conn:
                conn.execute('DELETE FROM file_cache WHERE file_hash = ?', (file_hash,))
    
    def refresh_cache(self, file_path: str, extend_seconds: int) -> bool:
        """延长缓存有效期
        
        Args:
            file_path: 文件路径
            extend_seconds: 延长的秒数
            
        Returns:
            是否成功延长缓存时间
        """
        file_hash = self._calculate_file_hash(file_path)
        current_time = int(time.time())
        
        with self.lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    'SELECT cache_expired_time FROM file_cache WHERE file_hash = ?',
                    (file_hash,)
                )
                row = cursor.fetchone()
                
                if not row:
                    return False
                
                new_expired_time = current_time + extend_seconds
                conn.execute('''
                    UPDATE file_cache 
                    SET cache_expired_time = ?, update_time = ?
                    WHERE file_hash = ?
                ''', (new_expired_time, current_time, file_hash))
                return True
    
    def clear_expired_cache(self) -> int:
        """清理所有过期的缓存记录
        
        Returns:
            清理的记录数量
        """
        current_time = int(time.time())
        
        with self.lock:
            with self._get_conn() as conn:
                cursor = conn.execute('''
                    DELETE FROM file_cache 
                    WHERE cache_expired_time IS NOT NULL 
                    AND cache_expired_time < ?
                ''', (current_time,))
                return cursor.rowcount
    
    def get_cache_by_tag(self, cache_tag: str) -> list:
        """根据缓存标签获取缓存记录
        
        Args:
            cache_tag: 缓存标签
            
        Returns:
            缓存记录列表
        """
        with self.lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    'SELECT * FROM file_cache WHERE cache_tag = ?',
                    (cache_tag,)
                )
                rows = cursor.fetchall()
                return [{
                    'file_hash': row[0],
                    'file_path': row[1],
                    'messages': self._deserialize_messages(row[2]),
                    'cache_tag': row[3],
                    'cache_expired_time': row[4],
                    'cache_message': row[5],
                    'create_time': row[6],
                    'update_time': row[7]
                } for row in rows]
    def get_cache_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """获取文件的完整缓存信息
        
        Args:
            file_path: 文件路径
            
        Returns:
            缓存记录字典，未命中返回None
        """
        if not os.path.exists(file_path):
            return None
            
        file_hash = self._calculate_file_hash(file_path)
        
        with self.lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    'SELECT * FROM file_cache WHERE file_hash = ?',
                    (file_hash,)
                )
                row = cursor.fetchone()
                
                if not row:
                    return None
                    
                # 检查缓存是否过期
                current_time = int(time.time())
                if row[4] and current_time > row[4]:  # cache_expired_time
                    self.remove_cache(file_hash)
                    return None
                    
                return {
                    'file_hash': row[0],
                    'file_path': row[1],
                    'messages': self._deserialize_messages(row[2]),
                    'cache_tag': row[3],
                    'cache_expired_time': row[4],
                    'cache_message': row[5],
                    'create_time': row[6],
                    'update_time': row[7]
                }
    def add_message(self, file_path: str, role: str, content: str, 
                   expire_seconds: Optional[int] = None) -> bool:
        """向缓存添加新消息"""
        cache_info = self.get_cache_info(file_path)
        if not cache_info:
            return False
            
        expired_time = (int(time.time()) + expire_seconds) if expire_seconds else None
        new_message = CacheMessage(role=role, content=content, expired_time=expired_time)
        
        messages = cache_info['messages']
        messages.append(new_message)
        
        self.set_cache(
            file_path=file_path,
            messages=messages,
            cache_tag=cache_info['cache_tag'],
            expire_seconds=cache_info['cache_expired_time'],
            cache_message=cache_info['cache_message']
        )
        return True
    
    def clear_expired_messages(self, file_path: str) -> int:
        """清理指定文件的过期消息"""
        cache_info = self.get_cache_info(file_path)
        if not cache_info:
            return 0
            
        original_count = len(cache_info['messages'])
        valid_messages = [msg for msg in cache_info['messages'] if not msg.is_expired()]
        
        if len(valid_messages) < original_count:
            self.set_cache(
                file_path=file_path,
                messages=valid_messages,
                cache_tag=cache_info['cache_tag'],
                expire_seconds=cache_info['cache_expired_time'],
                cache_message=cache_info['cache_message']
            )
            
        return original_count - len(valid_messages)