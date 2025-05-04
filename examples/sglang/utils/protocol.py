from pydantic import BaseModel
from typing import Dict, Any

class SGLangGenerateRequest(BaseModel):
    # Wrapper around the GenerateReqInput which is the input to the SGLang engine
    request_id: str
    input_ids: list[int]
    sampling_params: dict

class MyRequestOutput(BaseModel):
    text: Dict[str, Any]
