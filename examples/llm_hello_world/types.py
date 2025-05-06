from pydantic import BaseModel


class ChatRequest(BaseModel):
    messages: list[dict[str, str]]
    model: str


class ChatResponse(BaseModel):
    text: str


class RouteRequest(BaseModel):
    model: str
    text: str


class RouteResponse(BaseModel):
    text: str


class GenerateRequest(BaseModel):
    text: str
    model: str


class GenerateResponse(BaseModel):
    text: str
