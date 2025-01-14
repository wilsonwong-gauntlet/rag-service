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
    model="gpt-4-turbo-preview",
    temperature=0.7,
    openai_api_key=os.getenv('OPENAI_API_KEY')
)

# Define prompt template
RESPONSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are generating responses in a chat application. If there are previous messages, use them to match the user's style. If there are no previous messages, provide a natural, knowledgeable response to the current message.

Guidelines:
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