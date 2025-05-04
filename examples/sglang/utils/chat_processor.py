from sglang.srt.server_args import ServerArgs
from typing import Union
from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.openai_api.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
)
from sglang.srt.openai_api.adapter import v1_chat_generate_request, v1_generate_request
from utils.tokenizer_manager import DynamoTokenizerManager

class ProcessMixIn:
    """
    Mixin for pre and post processing for SGLang
    """

    engine_args: ServerArgs
    chat_processor: "ChatProcessor | None"
    completions_processor: "CompletionsProcessor | None"

    def __init__(self):
        # TODO: change name to PatchedSGLangTokenizerManager
        self.tokenizer_manager = DynamoTokenizerManager(self.engine_args)
        print("tokenizer manager initialized?")

    def _get_processor(
            self, raw_request: Union[ChatCompletionRequest, CompletionRequest]
    ):
        return (
            self.chat_processor
            if isinstance(raw_request, ChatCompletionRequest)
            else self.completions_processor
        )
    
    async def _parse_raw_request(
            self, raw_request: Union[ChatCompletionRequest, CompletionRequest]
    ): 
        processor = self._get_processor(raw_request)
        if processor is None:
            raise RuntimeError("Processor has not been initialized")
        preprocess_result, sampling_params = await processor.preprocess(raw_request, self.tokenizer_manager)

        return preprocess_result, sampling_params
    
class ChatProcessor:
    def parse_raw_request(self, raw_request: ChatCompletionRequest) -> ChatCompletionRequest:
        return ChatCompletionRequest.model_validate(raw_request)

    async def preprocess(self, raw_request: ChatCompletionRequest, tokenizer_manager: DynamoTokenizerManager) -> GenerateReqInput:
        chat_generate_req_input, _ = v1_chat_generate_request([raw_request], tokenizer_manager)
        return chat_generate_req_input, chat_generate_req_input.sampling_params

class CompletionsProcessor:
    def parse_raw_request(self, raw_request: CompletionRequest) -> CompletionRequest:
        return CompletionRequest.model_validate(raw_request)

    # the v1_generate_request does not tokenize under the hood so we return the TokenizedGenerateReqInput along with sampling params as a dict
    # because the sgl.Engine runs SamplingParams.__init__() 
    async def preprocess(self, raw_request: CompletionRequest, tokenizer_manager: DynamoTokenizerManager) -> TokenizedGenerateReqInput:
        generate_req_input, _ = v1_generate_request([raw_request])
        tokenized_generate_req_input = await tokenizer_manager._tokenize_one_request(generate_req_input)

        return tokenized_generate_req_input, generate_req_input.sampling_params
