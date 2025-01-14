from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import List
import os

llm = ChatOpenAI(
    model="gpt-4-turbo-preview",
    temperature=0.7,
    openai_api_key=os.getenv('OPENAI_API_KEY')
)

RESPONSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Generate a response based on the user's previous messages.
    Use the provided message history to understand their communication style and knowledge.
    The response should be natural and contextually appropriate."""),
    ("human", "Previous messages:\n{context_messages}\n\nCurrent message to respond to: {current_message}\n\nGenerate response:"),
])

async def generate_contextual_response(
    current_message: str,
    context_messages: List[str]
) -> str:
    context_str = "\n".join(context_messages)
    
    response = await llm.ainvoke(
        RESPONSE_PROMPT.format(
            context_messages=context_str,
            current_message=current_message
        )
    )
    
    return response.content