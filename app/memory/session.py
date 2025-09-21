import redis.asyncio as redis
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from app.config import settings


class SessionManager:
    def __init__(self):
        self.redis_client = redis.from_url(settings.redis_url)

    def _get_session_key(self, session_id: str) -> str:
        return f"session:{session_id}"

    def _get_history_key(self, session_id: str) -> str:
        return f"history:{session_id}"

    async def store_message(self, session_id: str, message_type: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        history_key = self._get_history_key(session_id)

        message = {
            "type": message_type,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }

        await self.redis_client.lpush(history_key, json.dumps(message))
        await self.redis_client.expire(history_key, 86400)  # Expire after 1 day

    async def get_session_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        history_key = self._get_history_key(session_id)

        messages = await self.redis_client.lrange(history_key, 0, limit - 1)
        return [json.loads(msg) for msg in messages]

    async def get_context(self, session_id: str, max_messages: int = 10) -> str:
        history = await self.get_session_history(session_id, max_messages)

        context_parts = []
        for message in reversed(history):  # Reverse to get chronological order
            if message["type"] == "question":
                context_parts.append(f"User: {message['content']}")
            elif message["type"] == "answer":
                context_parts.append(f"Assistant: {message['content']}")

        return "\n".join(context_parts)

    async def store_context(self, session_id: str, key: str, value: Any, expire_seconds: int = 3600):
        session_key = self._get_session_key(session_id)
        await self.redis_client.hset(session_key, key, json.dumps(value))
        await self.redis_client.expire(session_key, expire_seconds)

    async def get_context_value(self, session_id: str, key: str) -> Optional[Any]:
        session_key = self._get_session_key(session_id)
        value = await self.redis_client.hget(session_key, key)

        if value:
            return json.loads(value)
        return None

    async def clear_session(self, session_id: str):
        session_key = self._get_session_key(session_id)
        history_key = self._get_history_key(session_id)

        await self.redis_client.delete(session_key)
        await self.redis_client.delete(history_key)