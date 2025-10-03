# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from sglang.srt.parser.conversation import chat_templates

logger = logging.getLogger(__name__)


def clean_addcriterion(text: str) -> str:
    """
    Removes the addCriterion prefix from the text output.

    To prevent addCriterion from appearing in outputs, an assistant placeholder must be added to the conversation.
    However, adding the assistant placeholder causes subsequent requests to fail with shape mismatch errors on the engine side.
    The root cause is still under investigation, so this temporary workaround is in place to maintain functionality.
    """

    if text.startswith(" addCriterion"):
        cleaned_text = text[13:].lstrip()  # 12 = len(" addCriterion")
        logger.debug(
            f"🛠️ HACK: Removed ' addCriterion' prefix: '{text[:20]}...' -> '{cleaned_text[:20]}...'"
        )
        return cleaned_text

    if text.startswith("addCriterion"):
        cleaned_text = text[12:].lstrip()  # 11 = len("addCriterion")
        logger.debug(
            f"🛠️ HACK: Removed 'addCriterion' prefix: '{text[:20]}...' -> '{cleaned_text[:20]}...'"
        )
        return cleaned_text

    return text


def multimodal_request_to_sglang(raw_request, tokenizer, chat_template):
    conv = chat_templates[chat_template].copy()
    conv.messages = []

    # Convert messages into SGLang conversation
    for msg in raw_request.messages:
        if msg.role == "system":
            conv.system_message = msg.content
        elif msg.role == "user":
            text_parts = []
            for part in msg.content:
                if part.type == "text":
                    text_parts.append(part.text)
                elif part.type == "image_url":
                    text_parts.append(conv.image_token)
            conv.append_message(conv.roles[0], " ".join(text_parts))
        elif msg.role == "assistant":
            conv.append_message(conv.roles[1], msg.content)

    logger.debug(f"conv: {conv}")

    # Tokenize and prepare input_ids
    processed = tokenizer(text=conv.get_prompt(), return_tensors="pt")
    input_ids = processed["input_ids"][0].tolist()

    # Build the SGLang request dict
    sglang_request = {
        "model": raw_request.model,
        "token_ids": input_ids,
        "batch_token_ids": None,
        "stop_conditions": {"max_tokens": raw_request.max_tokens or None},
        "sampling_options": {"temperature": raw_request.temperature or 0.7},
        "eos_token_ids": [tokenizer.eos_token_id],
        "annotations": [],
        "stream": raw_request.stream if raw_request.stream is not None else False,
    }

    return sglang_request


def detokenize_sglang_response(response_data, tokenizer):
    """
    Detokenize SGLang response token IDs to text.

    Args:
        response_data: Dictionary containing token_ids and other response data
        tokenizer: The tokenizer to use for detokenization

    Returns:
        String containing the detokenized text, empty string if no tokens
    """
    try:
        # Handle Annotated objects from Dynamo (following vLLM-like pattern)
        if hasattr(response_data, "data"):
            try:
                import json

                raw_data = response_data.data
                # Handle callable data method
                if callable(raw_data):
                    raw_data = raw_data()
                response_data = (
                    json.loads(raw_data) if isinstance(raw_data, str) else raw_data
                )
            except (json.JSONDecodeError, AttributeError):
                try:
                    raw_data = response_data.data
                    if callable(raw_data):
                        raw_data = raw_data()
                    response_data = {"text": str(raw_data), "finished": False}
                except Exception:
                    response_data = {"text": str(response_data), "finished": False}

        # Ensure response_data is a dictionary
        if not isinstance(response_data, dict):
            return clean_addcriterion(str(response_data))

        # Get text content - detokenize if needed
        if "text" in response_data and response_data["text"]:
            return clean_addcriterion(response_data["text"])
        elif "token_ids" in response_data and response_data["token_ids"]:
            token_ids = response_data["token_ids"]
            if isinstance(token_ids, list) and token_ids:
                # Detokenize token IDs to get text
                text_content = tokenizer.decode(token_ids, skip_special_tokens=True)
                logger.debug(
                    f"Detokenized {len(token_ids)} tokens to: '{text_content}'"
                )
                return clean_addcriterion(text_content)

        # Return empty string if no content to detokenize
        return ""

    except Exception as e:
        logger.error(f"Failed to detokenize response: {e}")
        return f"[Detokenization error: {e}]"


def process_sglang_stream_response(response_data, tokenizer, accumulated_text=""):
    """
    Process a single SGLang streaming response with efficient detokenization.

    Args:
        response_data: Dictionary containing SGLang response data
        tokenizer: The tokenizer to use for detokenization
        accumulated_text: Previously accumulated text for context

    Returns:
        Tuple of (text_content, updated_accumulated_text, is_finished)
    """
    try:
        # Handle Annotated objects from Dynamo (following vLLM-like pattern)
        if hasattr(response_data, "data"):
            try:
                import json

                raw_data = response_data.data
                # Handle callable data method
                if callable(raw_data):
                    raw_data = raw_data()
                response_data = (
                    json.loads(raw_data) if isinstance(raw_data, str) else raw_data
                )
            except (json.JSONDecodeError, AttributeError):
                try:
                    raw_data = response_data.data
                    if callable(raw_data):
                        raw_data = raw_data()
                    response_data = {"text": str(raw_data), "finished": False}
                except Exception:
                    response_data = {"text": str(response_data), "finished": False}

        # Ensure response_data is a dictionary
        if not isinstance(response_data, dict):
            response_data = {"text": str(response_data), "finished": False}

        # Detokenize the current response
        text_content = detokenize_sglang_response(response_data, tokenizer)

        # Update accumulated text
        new_accumulated = accumulated_text + text_content

        # Check if this is the final response
        is_finished = response_data.get("finished", False) or response_data.get(
            "finish_reason"
        )

        return text_content, new_accumulated, is_finished

    except Exception as e:
        logger.error(f"Error processing SGLang stream response: {e}")
        return f"[Processing error: {e}]", accumulated_text, True
