from celery import Celery
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import os
import httpx
import tempfile
import asyncio
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

load_dotenv()

# Initialize Celery
celery_app = Celery(
    'message_processor',
    broker=os.getenv('REDIS_URL'),
    backend=os.getenv('REDIS_URL')
)

embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv('OPENAI_API_KEY'), 
    model="text-embedding-3-large"
)

@celery_app.task(bind=True, max_retries=3)
def process_message(self, message_data: dict):
    try:
        metadata = {
            "messageId": message_data["id"],
            "workspaceId": message_data["workspaceId"],
            "userId": message_data["userId"],
            "channelId": message_data["channelId"]
        }
        
        # Create embedding and store in Pinecone
        texts = [message_data["content"]]
        vector_store = PineconeVectorStore.from_texts(
            texts,
            embeddings,
            index_name=os.getenv('PINECONE_INDEX'),
            metadatas=[metadata]
        )
        
        return {"status": "success", "messageId": message_data["id"]}
        
    except Exception as e:
        self.retry(exc=e, countdown=60)  # Retry after 60 seconds

def download_file_sync(url: str, file_path: str):
    with httpx.Client() as client:
        response = client.get(url)
        response.raise_for_status()
        with open(file_path, 'wb') as f:
            f.write(response.content)

def get_document_loader(file_path: str, file_type: str):
    if file_type == "application/pdf":
        return PyPDFLoader(file_path)
    else:
        return TextLoader(file_path)

def chunk_document(texts: List[str]) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_text("\n\n".join(texts))

@celery_app.task(bind=True, max_retries=3)
def process_document(
    self,
    document_id: str,
    workspace_id: str,
    file_url: str,
    file_name: str,
    file_type: str
) -> Dict[str, Any]:
    try:
        # Create a temporary file to store the download
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp_file:
            # Download file
            download_file_sync(file_url, tmp_file.name)
            
            # Load and process document
            loader = get_document_loader(tmp_file.name, file_type)
            documents = loader.load()
            
            # Extract text content
            texts = [doc.page_content for doc in documents]
            
            # Chunk the document
            chunks = chunk_document(texts)
            
            # Prepare metadata for each chunk
            metadatas = [{
                "documentId": document_id,
                "workspaceId": workspace_id,
                "fileName": file_name,
                "chunkIndex": i
            } for i in range(len(chunks))]
            
            # Store in vector database
            vector_store = PineconeVectorStore.from_texts(
                chunks,
                embeddings,
                index_name=os.getenv('PINECONE_INDEX'),
                metadatas=metadatas
            )
            
            # Cleanup temporary file
            os.unlink(tmp_file.name)
            
            return {
                "success": True,
                "documentId": document_id,
                "chunks": len(chunks)
            }
            
    except Exception as e:
        return {
            "success": False,
            "documentId": document_id,
            "error": str(e)
        }

# def send_callback_sync(url: str, token: str, data: Dict[str, Any]):
#     with httpx.Client() as client:
#         try:
#             client.post(
#                 url,
#                 json=data,
#                 headers={"x-callback-token": token},
#                 timeout=10.0
#             )
#         except Exception as e:
#             print(f"Callback failed: {str(e)}")

# @celery_app.task(bind=True, max_retries=3)
# def delete_document(
#     self,
#     document_id: str,
#     workspace_id: str,
#     callback_url: Optional[str] = None,
#     callback_token: Optional[str] = None
# ) -> Dict[str, Any]:
#     try:
#         # Initialize vector store
#         vector_store = PineconeVectorStore.from_existing_index(
#             index_name=os.getenv('PINECONE_INDEX'),
#             embedding=embeddings
#         )
#         
#         # Delete all chunks for this document
#         vector_store.delete(
#             filter={
#                 "$and": [
#                     {"document_id": {"$eq": document_id}},
#                     {"workspace_id": {"$eq": workspace_id}}
#                 ]
#             }
#         )
#         
#         result = {
#             "success": True
#         }
#         
#         # Send callback if URL provided
#         if callback_url and callback_token:
#             callback_data = {
#                 "documentId": document_id,
#                 "status": "DELETED"
#             }
#             send_callback_sync(callback_url, callback_token, callback_data)
#         
#         return result
#         
#     except Exception as e:
#         error_result = {
#             "success": False,
#             "error": str(e)
#         }
#         
#         # Send error callback if URL provided
#         if callback_url and callback_token:
#             callback_data = {
#                 "documentId": document_id,
#                 "status": "FAILED",
#                 "error": str(e)
#             }
#             send_callback_sync(callback_url, callback_token, callback_data)
#         
#         return error_result
