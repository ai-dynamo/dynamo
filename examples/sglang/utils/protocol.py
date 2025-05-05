from typing import Any, Dict

from pydantic import BaseModel


class SGLangGenerateRequest(BaseModel):
    # Wrapper around the GenerateReqInput which is the input to SGLang engine
    request_id: str
    input_ids: list[int]
    sampling_params: dict


class MyRequestOutput(BaseModel):
    text: Dict[str, Any]
