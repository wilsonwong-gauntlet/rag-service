from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class MessageEvent(BaseModel):
    id: str
    content: str
    channelId: str
    workspaceId: str
    userId: str
    userName: str
    channelName: str
    createdAt: datetime

class SearchQuery(BaseModel):
    query: str
    workspaceId: str
    channelId: Optional[str] = None
    receiverId: str
    limit: int = 5

class SearchResult(BaseModel):
    content: str
    messageId: str
    userName: str
    channelName: str
    timestamp: datetime

class SearchResponse(BaseModel):
    messages: List[SearchResult]
