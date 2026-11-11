# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate verified TRT-LLM grammar compilation rejections into client errors."""

import re

from dynamo.llm.exceptions import InvalidArgument

# Temporary workaround until TRT-LLM preserves a typed error or code across the
# executor boundary. The guided-decoding prefix alone also covers engine bugs.
# Recognize only xgrammar's unsupported schema type and unclosed character class.
_COMPILATION_REJECTION = re.compile(
    r"^Guided decoding error: (?:\[\d{2}:\d{2}:\d{2}\] )?(?:[^\n]*[/\\])?"
    r'(?:json_schema_converter\.cc:\d+: Unsupported type "[^"\n]*"'
    r"|regex_converter\.cc:\d+: Regex parsing error at position \d+: Unclosed '\[')$"
)


def guided_decoding_error_to_invalid_argument(error_msg: str) -> InvalidArgument | None:
    """Classify known compiler diagnostics; leave ambiguous failures as server errors.

    Call only before output begins. Compiler locations may include an absolute
    path, and xgrammar may append a stack trace after the diagnostic line.
    """
    if _COMPILATION_REJECTION.fullmatch(error_msg.partition("\n")[0]):
        return InvalidArgument(error_msg)
    return None
