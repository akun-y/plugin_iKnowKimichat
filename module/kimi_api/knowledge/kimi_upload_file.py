from pathlib import Path
import time
from typing import Dict, List, Optional, Any

import httpx
from loguru import logger
from openai import OpenAI
def upload_files(client:OpenAI, files: List[str], cache_tag: Optional[str] = None) -> List[Dict[str, Any]]:
        """通过文件或缓存构建message,作为chat上下文
        
        Args:
            client: OpenAI对象
            files: 文件列表
            cache_tag: 缓存标签
        Returns:
            List[Dict[str, Any]]: 包含消息的列表
        """
        messages = []
    
        for file in files:
            file_object = client.files.create(file=Path(file), purpose="file-extract")
            file_content = client.files.content(file_id=file_object.id).text
            messages.append({   
                "role": "system",
                "content": file_content,
                "expired_time": None
            })
            logger.warning(f"[KimiUploadFile] 上传文件: {file} 成功")
    
        if cache_tag:
            r = httpx.post(f"{client.base_url}caching",
                        headers={
                            "Authorization": f"Bearer {client.api_key}",
                        },
                        json={
                            "model": "moonshot-v1",
                            "messages": messages,
                            "ttl": 300,
                            "tags": [cache_tag],
                        })
    
            if r.status_code != 200:
                raise Exception(r.text)
            logger.warning(f"[KimiUploadFile] 上传文件设置为缓存成功")
            # 省去文件传输及token
            return [{
                "role": "cache",
                "content": f"tag={cache_tag};reset_ttl=300",
                "expired_time": int(time.time()) + 300
            }]
        else:
            return messages
    