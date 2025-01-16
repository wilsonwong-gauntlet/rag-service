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
    messageId: Optional[str] = None
    documentId: Optional[str] = None
    documentName: Optional[str] = None

class AIResponse(BaseModel):
    response: str
    confidence: float
    sourceMessages: List[SearchResult]

class ProcessDocumentRequest(BaseModel):
    documentId: str
    workspaceId: str
    fileUrl: str
    fileName: str
    fileType: str

class ProcessDocumentResponse(BaseModel):
    success: bool
    documentId: str
    chunks: Optional[int] = None
    error: Optional[str] = None

# class DeleteDocumentRequest(BaseModel):
#     documentId: str
#     workspaceId: str
#     callbackUrl: Optional[str] = None
#     callbackToken: Optional[str] = None

# class DeleteDocumentResponse(BaseModel):
#     success: bool
#     error: Optional[str] = None

class KnowledgeBaseRequest(BaseModel):
    query: str
    workspaceId: str
    limit: int = 5