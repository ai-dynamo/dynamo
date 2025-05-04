from dynamo.sdk.lib.config import ServiceConfig
import os
import json

import argparse
from dynamo.sdk.lib.config import ServiceConfig
from sglang.srt.server_args import ServerArgs

# def parse_sglang_args(service_name, prefix) -> ServerArgs:
#     config = ServiceConfig.get_instance()
#     sglang_args = config.as_args(service_name, prefix=prefix)
#     parser = argparse.ArgumentParser()
#     # add future dynamo arguments here
#     ServerArgs.add_cli_args(parser)
#     args = parser.parse_args(sglang_args)
#     return ServerArgs.from_cli_args(args)


# # Set test config in env var
# test_config = {
#     "test_service": {
#         "model-path": "/path/to/model",
#         "port": "8080"
#     }
# }
# os.environ["DYNAMO_SERVICE_CONFIG"] = json.dumps(test_config)

# # Parse args
# args = parse_sglang_args("test_service", "")

# # Print results
# print(f"Parsed args:")
# print(f"Model path: {args.model_path}")
# print(f"Port: {args.port}")

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.openai_api.protocol import ChatCompletionRequest, CompletionRequest
# from utils.chat_processor import ChatProcessor, CompletionsProcessor, DynamoTokenizerManager
from sglang.srt.server_args import ServerArgs
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    TokenizedGenerateReqInput,
    EmbeddingReqInput,
    TokenizedEmbeddingReqInput,
    SessionParams,
    SamplingParams,
)
from typing import Dict, List, Optional, Union
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.openai_api.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ChatCompletionResponse,
    CompletionResponse,
)
from sglang.srt.openai_api.adapter import v1_chat_generate_request, v1_generate_request

class DynamoTokenizerManager(TokenizerManager):
    # TODO: move to rust!
    """
    This class acts as a patch for the `TokenizerManager` class in SGLang. Because
    Dynamo has it's own communication primatives - we don't need to use the
    ZMQ initialization and communication. We also don't currently support multimodel
    inputs or embedding APIs.

    Ideally much of this (if not all) can and should get handled in rust land very soon. But 
    in order to use sglang helper functions like `v1_chat_generate_request` we need to be able to pass
    in a `TokenizerManager` instance (even though it's not expressly typed as such).
    """

    def __init__(self, server_args: ServerArgs):
        self.server_args = server_args

        self.model_path = server_args.model_path
        self.served_model_name = server_args.served_model_name
        self.model_config = ModelConfig(
            server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
            context_length=server_args.context_length,
            model_override_args=server_args.json_model_override_args,
            is_embedding=server_args.is_embedding,
            enable_multimodal=server_args.enable_multimodal,
            dtype=server_args.dtype,
            quantization=server_args.quantization,
        )
        self.tokenizer = get_tokenizer(
            tokenizer_name=server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
        )

        self.is_generation = self.model_config.is_generation
        self.context_len = self.model_config.context_len


    async def _tokenize_one_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
    ):
        """Tokenize one request."""
        # Tokenize
        input_embeds = None
        input_text = obj.text
        if obj.input_embeds is not None:
            if not self.server_args.disable_radix_cache:
                raise ValueError(
                    "input_embeds is provided while disable_radix_cache is False. "
                    "Please add `--disable-radix-cache` when you launch the server "
                    "if you want to use input_embeds as inputs."
                )
            input_embeds = obj.input_embeds
            input_ids = obj.input_ids
        elif obj.input_ids is not None:
            input_ids = obj.input_ids
        else:
            if self.tokenizer is None:
                raise ValueError(
                    "The engine initialized with skip_tokenizer_init=True cannot "
                    "accept text prompts. Please provide input_ids or re-initialize "
                    "the engine with skip_tokenizer_init=False."
                )
            input_ids = self.tokenizer.encode(input_text)

        self._validate_token_len(obj, input_ids)
        return self._create_tokenized_object(
            obj, input_text, input_ids, input_embeds
        )

    def _validate_token_len(
        self, obj: Union[GenerateReqInput, EmbeddingReqInput], input_ids: List[int]
    ) -> None:
        """Validates that the input token count and the requested token count doesn't exceed the model's context length."""

        input_token_num = len(input_ids) if input_ids is not None else 0
        # Check if input alone exceeds context length
        if input_token_num >= self.context_len:
            raise ValueError(
                f"The input ({input_token_num} tokens) is longer than the "
                f"model's context length ({self.context_len} tokens)."
            )

        # Check total tokens (input + max_new_tokens)
        max_new_tokens = obj.sampling_params.get("max_new_tokens")
        if (
            max_new_tokens is not None
            and (max_new_tokens + input_token_num) >= self.context_len
        ):
            total_tokens = max_new_tokens + input_token_num
            error_msg = (
                f"Requested token count exceeds the model's maximum context length "
                f"of {self.context_len} tokens. You requested a total of {total_tokens} "
                f"tokens: {input_token_num} tokens from the input messages and "
                f"{max_new_tokens} tokens for the completion. Please reduce the number "
                f"of tokens in the input messages or the completion to fit within the limit."
            )
            raise ValueError(error_msg)

    def _create_tokenized_object(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        input_text: str,
        input_ids: List[int],
        input_embeds: Optional[Union[List[float], None]] = None,
        image_inputs: Optional[Dict] = None,
    ) -> Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput]:
        """Create a tokenized request object from common parameters."""

        if self.is_generation:
            return_logprob = obj.return_logprob
            logprob_start_len = obj.logprob_start_len
            top_logprobs_num = obj.top_logprobs_num
            token_ids_logprob = obj.token_ids_logprob
            session_params = (
                SessionParams(**obj.session_params) if obj.session_params else None
            )

        sampling_params = SamplingParams(**obj.sampling_params)
        sampling_params.normalize(self.tokenizer)
        sampling_params.verify()

        # Build return object
        if isinstance(obj, GenerateReqInput):
            tokenized_obj = TokenizedGenerateReqInput(
                obj.rid,
                input_text,
                input_ids,
                image_inputs,
                sampling_params,
                return_logprob,
                logprob_start_len,
                top_logprobs_num,
                token_ids_logprob,
                obj.stream,
                bootstrap_host=obj.bootstrap_host,
                bootstrap_port=obj.bootstrap_port,
                bootstrap_room=obj.bootstrap_room,
                lora_path=obj.lora_path,
                input_embeds=input_embeds,
                session_params=session_params,
                custom_logit_processor=obj.custom_logit_processor,
                return_hidden_states=obj.return_hidden_states,
            )
        elif isinstance(obj, EmbeddingReqInput):
            tokenized_obj = TokenizedEmbeddingReqInput(
                obj.rid,
                input_text,
                input_ids,
                image_inputs,
                sampling_params,
            )

        return tokenized_obj

class ProcessMixIn:
    """
    Mixin for pre and post processing for SGLang
    """

    engine_args: ServerArgs
    chat_processor: "ChatProcessor | None"
    completions_processor: "CompletionsProcessor | None"

    def __init__(self):
        self.tokenizer_manager = DynamoTokenizerManager(self.engine_args)

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
        request = processor.parse_raw_request(raw_request)
        preprocess_result = await processor.preprocess(raw_request, self.tokenizer_manager)

        return preprocess_result

class ChatProcessor:
    # TODO: this and the vllm example should be called validate_raw_request
    def parse_raw_request(self, raw_request: ChatCompletionRequest) -> ChatCompletionRequest:
        return ChatCompletionRequest.model_validate(raw_request)

    async def preprocess(self, raw_request: ChatCompletionRequest, tokenizer_manager: DynamoTokenizerManager) -> GenerateReqInput:
        # v1_chat_generate_request tokenizes under the hood
        chat_generate_req_input, _ = v1_chat_generate_request([raw_request], tokenizer_manager)
        return chat_generate_req_input, chat_generate_req_input.sampling_params

class CompletionsProcessor:
        # TODO: this and the vllm example should be called validate_raw_request
    def parse_raw_request(self, raw_request: CompletionRequest) -> CompletionRequest:
        return CompletionRequest.model_validate(raw_request)

    async def preprocess(self, raw_request: CompletionRequest, tokenizer_manager: DynamoTokenizerManager) -> TokenizedGenerateReqInput:
        generate_req_input, _ = v1_generate_request([raw_request])
        # the v1_generate_request does not tokenize under the hood
        # due to this - we need to tokenize manually
        tokenized_generate_req_input = await tokenizer_manager._tokenize_one_request(generate_req_input)
        sampling_params = generate_req_input.sampling_params

        # we return the sampling params here so we can initialize the engine with it
        return tokenized_generate_req_input, sampling_params

import argparse
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from dynamo.sdk.lib.config import ServiceConfig
from sglang.srt.server_args import ServerArgs
from typing import Union
import sglang as sgl

SGLANG_TOKENIZERS = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

async def test_chat_processor():
    # Create test ServerArgs
    args = ServerArgs(
        model_path="Qwen/Qwen2.5-0.5B",
        served_model_name="Qwen2.5-0.5B", 
        trust_remote_code=True,
        skip_tokenizer_init=True,
        disable_cuda_graph=True,
    )

    # Create ProcessMixIn instance
    class TestProcessor(ProcessMixIn):
        def __init__(self, args):
            self.engine_args = args
            super().__init__()
            self.chat_processor = ChatProcessor()
            self.completions_processor = CompletionsProcessor()

    processor = TestProcessor(args)

    # Create test requests
    chat_request = ChatCompletionRequest(
        model="Qwen/Qwen2.5-0.5B",
        messages=[{"role": "user", "content": "Hello!"}]
    )

    completion_request = CompletionRequest(
        model="Qwen/Qwen2.5-0.5B",
        prompt="Hello",
        stream=True
    )

    chat_result, sp_chat = await processor._parse_raw_request(chat_request)
    completion_result, sp_completion = await processor._parse_raw_request(completion_request)

    engine = sgl.Engine(server_args=args)

    print("------------completion_result------------")

    g = await engine.async_generate(
        input_ids=completion_result.input_ids,
        sampling_params=sp_completion,
        stream=True
    )
    async for result in g:
        res = processor.tokenizer_manager.tokenizer.batch_decode(result.get("output_ids"))
        print(res)
    

    print("-----------chat_result------------")

    g = await engine.async_generate(
        input_ids=chat_result.input_ids,
        sampling_params=sp_chat,
        stream=True
    )
    async for result in g:
        res = processor.tokenizer_manager.tokenizer.batch_decode(result.get("output_ids"))
        print(res)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_chat_processor())
