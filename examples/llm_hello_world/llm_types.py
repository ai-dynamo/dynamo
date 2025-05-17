from pydantic import BaseModel


class ChatRequest(BaseModel):
    text: str

