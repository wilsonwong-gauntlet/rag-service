from celery import Celery
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv

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
