import logging
from datetime import datetime
from typing import List
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_service.log'),
        logging.StreamHandler()  # This will print to console as well
    ]
)

logger = logging.getLogger('rag_service')

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    openai_api_key=os.getenv('OPENAI_API_KEY')
)

# Define prompt template
RESPONSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are generating responses in a chat application. Respond as if you were the actual user.If there are previous messages, use them to match the user's style. If there are no previous messages, provide a natural, knowledgeable response to the current message.

Guidelines:
- You are pretending to be a user in a chat application.
- Provide direct, informative responses
- If it's a technical topic, give a balanced, knowledgeable answer
- If it's an opinion question, give a thoughtful perspective
- Keep responses concise and conversational
- Never mention being AI or ask for clarification
- Never apologize or express uncertainty about responding"""),
    ("human", """Previous messages (if any):
{context_messages}

Current message: {current_message}

Generate a natural response:"""),
])

# Define knowledge base prompt template
KNOWLEDGE_BASE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an AI that creates extremely concise document summaries. Your responses must follow this exact format:

1. One paragraph summary (2-3 sentences max)
2. 3-4 key bullet points
3. Optional single-sentence conclusion

Guidelines:
- Entire response must be under 150 words
- Use simple, clear language
- Focus only on the most important information
- If documents lack relevant info, respond with a single sentence stating this
- No additional sections or formatting allowed"""),
    ("human", """Available document excerpts:
{context_messages}

Question: {current_message}

Provide a concise summary following the required format:"""),
])

async def generate_contextual_response(
    current_message: str,
    context_messages: List[str]
) -> str:
    request_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    
    logger.info(f"[{request_id}] Generating response for message: {current_message}")
    logger.info(f"[{request_id}] Context messages available: {len(context_messages)}")
    
    if context_messages:
        logger.info(f"[{request_id}] Context messages:")
        for i, msg in enumerate(context_messages):
            logger.info(f"[{request_id}] Context[{i}]: {msg}")
    else:
        logger.info(f"[{request_id}] No context messages available")

    context_str = "\n".join(context_messages) if context_messages else "No previous messages available."
    
    try:
        logger.info(f"[{request_id}] Calling LLM with prompt...")
        response = await llm.ainvoke(
            RESPONSE_PROMPT.format(
                context_messages=context_str,
                current_message=current_message
            )
        )
        logger.info(f"[{request_id}] Generated response: {response.content}")
        return response.content
        
    except Exception as e:
        logger.error(f"[{request_id}] Error generating response: {str(e)}", exc_info=True)
        raise

async def generate_knowledge_base_response(
    query: str,
    document_contexts: List[str]
) -> str:
    request_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    
    logger.info(f"[{request_id}] Generating knowledge base response for query: {query}")
    logger.info(f"[{request_id}] Document contexts available: {len(document_contexts)}")
    
    if document_contexts:
        logger.info(f"[{request_id}] Document contexts:")
        for i, ctx in enumerate(document_contexts):
            logger.info(f"[{request_id}] Context[{i}]: {ctx}")
    else:
        logger.info(f"[{request_id}] No document contexts available")

    context_str = "\n\n---\n\n".join(document_contexts) if document_contexts else "No relevant documents found."
    
    try:
        logger.info(f"[{request_id}] Calling LLM with knowledge base prompt...")
        response = await llm.ainvoke(
            KNOWLEDGE_BASE_PROMPT.format(
                context_messages=context_str,
                current_message=query
            )
        )
        logger.info(f"[{request_id}] Generated response: {response.content}")
        return response.content
        
    except Exception as e:
        logger.error(f"[{request_id}] Error generating response: {str(e)}", exc_info=True)
        raise