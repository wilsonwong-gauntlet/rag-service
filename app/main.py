from fastapi import FastAPI, HTTPException
from .schemas import (
    MessageEvent, SearchQuery, SearchResult, AIResponse, 
    GenerateRequest, ProcessDocumentRequest, ProcessDocumentResponse,
    DeleteVectorsRequest, DeleteVectorsResponse,
    KnowledgeBaseRequest
)
from .processor import process_message, embeddings, process_document, delete_vectors
from .llm import generate_contextual_response, generate_knowledge_base_response
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Slack RAG Service")

@app.post("/message-event")
async def handle_message_event(message: MessageEvent):
    """Queue a message for processing"""
    try:
        # Queue the message for processing
        process_message.delay(message.dict())
        return {"status": "queued", "messageId": message.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_messages(query: SearchQuery):
    """Search for relevant messages"""
    try:
        # Build filter based on workspace and sender
        filter_dict = {
            "workspaceId": query.workspaceId,
            "userId": query.receiverId
        }

        vector_store = PineconeVectorStore.from_existing_index(
            index_name=os.getenv('PINECONE_INDEX'),
            embedding=embeddings
        )
        
        results = vector_store.similarity_search(
            query.query,
            k=query.limit,
            filter=filter_dict
        )
        
        messages = [
            SearchResult(
                content=doc.page_content,
                messageId=doc.metadata["messageId"]
            ) for doc in results
        ]
        
        return {"messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate", response_model=AIResponse)
async def generate_response(query: GenerateRequest):
    """Generate an AI response using RAG context"""
    try:
        # Build filter based on workspace and sender
        filter_dict = {
            "workspaceId": query.workspaceId,
            "userId": query.receiverId
        }

        vector_store = PineconeVectorStore.from_existing_index(
            index_name=os.getenv('PINECONE_INDEX'),
            embedding=embeddings
        )
        
        # Get relevant context messages
        results = vector_store.similarity_search_with_score(
            query.query,
            k=query.limit,
            filter=filter_dict
        )
        
        # Convert to source messages
        source_messages = [
            SearchResult(
                content=doc.page_content,
                messageId=doc.metadata["messageId"]
            ) for doc, score in results
        ]
        
        # Generate response using LLM
        response = await generate_contextual_response(
            current_message=query.query,
            context_messages=[msg.content for msg in source_messages]
        )
        
        return AIResponse(
            response=response,
            confidence=1.0 if results else 0.5,
            sourceMessages=source_messages
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-document", response_model=ProcessDocumentResponse)
async def handle_process_document(request: ProcessDocumentRequest):
    """Process a document for RAG"""
    try:
        # Queue the document processing task
        task = process_document.delay(
            request.documentId,
            request.workspaceId,
            request.fileUrl,
            request.fileName,
            request.fileType
        )
        result = task.get()  # Wait for the result
        return result
    except Exception as e:
        return ProcessDocumentResponse(
            success=False,
            documentId=request.documentId,
            error=str(e)
        ) 

# @app.post("/delete-document", response_model=DeleteDocumentResponse)
# async def handle_delete_document(request: DeleteDocumentRequest):
#     """Delete a document and its chunks from the vector store"""
#     try:
#         # Queue the delete task
#         task = delete_document.delay(
#             request.documentId,
#             request.workspaceId,
#             request.callbackUrl,
#             request.callbackToken
#         )
#         result = task.get()  # Wait for the result
#         return result
#     except Exception as e:
#         return DeleteDocumentResponse(
#             success=False,
#             error=str(e)
#         ) 

@app.post("/knowledge-base/generate", response_model=AIResponse)
async def handle_knowledge_base_generate(query: KnowledgeBaseRequest):
    """Generate an AI response using workspace documents"""
    try:
        # Build filter based on workspace only
        filter_dict = {
            "workspaceId": query.workspaceId
        }

        vector_store = PineconeVectorStore.from_existing_index(
            index_name=os.getenv('PINECONE_INDEX'),
            embedding=embeddings
        )
        
        # Get relevant context from documents
        results = vector_store.similarity_search_with_score(
            query.query,
            k=query.limit,
            filter=filter_dict
        )
        
        # Generate response using knowledge base LLM prompt
        response = await generate_knowledge_base_response(
            query=query.query,
            document_contexts=[doc.page_content for doc, score in results]
        )
        
        return AIResponse(
            response=response,
            confidence=1.0 if results else 0.5,
            sourceMessages=[
                SearchResult(
                    content=doc.page_content,
                    documentId=doc.metadata.get("documentId"),
                    documentName=doc.metadata.get("fileName")
                ) for doc, score in results
            ]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete-vectors", response_model=DeleteVectorsResponse)
async def handle_delete_vectors(request: DeleteVectorsRequest):
    """Delete vectors by their IDs, verifying workspace ownership"""
    try:
        task = delete_vectors.delay(
            request.vectorIds,
            request.workspaceId
        )
        result = task.get()  # Wait for the result
        return result
    except Exception as e:
        return DeleteVectorsResponse(
            success=False,
            deletedCount=0,
            error=str(e)
        ) 