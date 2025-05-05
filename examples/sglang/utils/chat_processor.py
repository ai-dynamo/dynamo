import logging
import time
import uuid
from typing import AsyncIterator, Union

from sglang.srt.managers.io_struct import GenerateReqInput, TokenizedGenerateReqInput
from sglang.srt.openai_api.adapter import v1_chat_generate_request, v1_generate_request
from sglang.srt.openai_api.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    CompletionRequest,
    DeltaMessage,
)
from sglang.srt.server_args import ServerArgs
from utils.protocol import MyRequestOutput
from utils.tokenizer_manager import DynamoTokenizerManager

logger = logging.getLogger(__name__)


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
        preprocess_result, sampling_params = await processor.preprocess(
            raw_request, self.tokenizer_manager
        )

        return preprocess_result, sampling_params


class ChatProcessor:
    def parse_raw_request(
        self, raw_request: ChatCompletionRequest
    ) -> ChatCompletionRequest:
        return ChatCompletionRequest.model_validate(raw_request)

    async def preprocess(
        self,
        raw_request: ChatCompletionRequest,
        tokenizer_manager: DynamoTokenizerManager,
    ) -> GenerateReqInput:
        chat_generate_req_input, _ = v1_chat_generate_request(
            [raw_request], tokenizer_manager
        )
        return chat_generate_req_input, chat_generate_req_input.sampling_params

    async def generate_stream_response(
        self,
        request: ChatCompletionRequest,
        sglang_generator: AsyncIterator,
        tokenizer_manager=None,
    ):
        """Converts SGLang stream outputs to OpenAI-compatible chat completion responses"""
        is_first = True
        stream_buffer = ""
        created = int(time.time())
        content_id = str(uuid.uuid4())

        async for response in sglang_generator:
            try:
                response_data = response.data()

                output = MyRequestOutput.model_validate_json(response_data)
                content = output.text

                if is_first:
                    is_first = False
                    choice = ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(role="assistant"),
                        finish_reason=None,
                    )
                    chunk = ChatCompletionStreamResponse(
                        id=content_id,
                        created=created,
                        choices=[choice],
                        model=request.model,
                    )
                    yield chunk.model_dump()

                output_ids = content.get("output_ids", [])
                meta_info = content.get("meta_info", {})

                if (
                    tokenizer_manager
                    and hasattr(tokenizer_manager, "tokenizer")
                    and tokenizer_manager.tokenizer
                ):
                    curr_text = tokenizer_manager.tokenizer.decode(output_ids)
                else:
                    curr_text = str(output_ids)

                delta = curr_text[len(stream_buffer) :]
                stream_buffer = curr_text

                if delta:
                    choice = ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(content=delta),
                        finish_reason=None,
                    )
                    chunk = ChatCompletionStreamResponse(
                        id=meta_info.get("id", content_id),
                        created=created,
                        choices=[choice],
                        model=request.model,
                    )
                    yield chunk.model_dump()

            except Exception as e:
                logger.error(f"Error processing response: {e}")
                error_choice = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(content=f"Error: {str(e)}"),
                    finish_reason="error",
                )
                error_chunk = ChatCompletionStreamResponse(
                    id=content_id,
                    created=created,
                    choices=[error_choice],
                    model=request.model,
                )
                yield error_chunk.model_dump()

        choice = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(),
            finish_reason="stop",
        )
        final_chunk = ChatCompletionStreamResponse(
            id=content_id,
            created=created,
            choices=[choice],
            model=request.model,
        )
        yield final_chunk.model_dump()


class CompletionsProcessor:
    def parse_raw_request(self, raw_request: CompletionRequest) -> CompletionRequest:
        return CompletionRequest.model_validate(raw_request)

    # the v1_generate_request does not tokenize under the hood so we return the TokenizedGenerateReqInput along with sampling params as a dict
    # because the sgl.Engine runs SamplingParams.__init__()
    async def preprocess(
        self, raw_request: CompletionRequest, tokenizer_manager: DynamoTokenizerManager
    ) -> TokenizedGenerateReqInput:
        generate_req_input, _ = v1_generate_request([raw_request])
        tokenized_generate_req_input = await tokenizer_manager._tokenize_one_request(
            generate_req_input
        )

        return tokenized_generate_req_input, generate_req_input.sampling_params
