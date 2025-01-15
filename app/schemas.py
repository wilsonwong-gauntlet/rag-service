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
    receiverId: str
    limit: int = 5

class GenerateRequest(BaseModel):
    query: str
    workspaceId: str
    receiverId: str
    limit: int = 5

class SearchResult(BaseModel):
    content: str
    messageId: str

class AIResponse(BaseModel):
    response: str
    confidence: float
    sourceMessages: List[SearchResult]