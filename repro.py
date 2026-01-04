import asyncio

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# ----------------------------
# Structured output schema
# ----------------------------
class AnswerSchema(BaseModel):
    question: str = Field(description="The original user question")
    answer: str = Field(description="The answer to the question")


# ----------------------------
# LLM setup
# ----------------------------
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",
    model="Qwen/Qwen2.5-7B-Instruct",
    temperature=0,
)

# ----------------------------
# Prompt
# ----------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a precise and concise AI assistant."),
        ("human", "Answer the following question:\n{question}"),
    ]
)

# ----------------------------
# Agent-like chain with structured output
# ----------------------------
chain = prompt | llm.with_structured_output(AnswerSchema)


# ----------------------------
# Async invocation
# ----------------------------
async def main():
    result = await chain.ainvoke({"question": "What is the capital of France?"})
    print(result)


asyncio.run(main())
