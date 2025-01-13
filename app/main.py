from fastapi import FastAPI, HTTPException
from .schemas import MessageEvent, SearchQuery, SearchResponse, SearchResult
from .processor import process_message, embeddings
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

@app.post("/search", response_model=SearchResponse)
async def search_messages(query: SearchQuery):
    """Search for relevant messages"""
    try:
        # Build filter based on workspace and optional channel
        filter_dict = {"workspaceId": query.workspaceId}
        if query.channelId:
            filter_dict["channelId"] = query.channelId

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
                messageId=doc.metadata["messageId"],
                userName=doc.metadata["userName"],
                channelName=doc.metadata["channelName"],
                timestamp=doc.metadata["timestamp"]
            ) for doc in results
        ]
        
        return SearchResponse(messages=messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)