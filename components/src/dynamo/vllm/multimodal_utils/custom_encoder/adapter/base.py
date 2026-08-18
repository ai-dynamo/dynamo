# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consumer-selected adapters for in-process custom vision encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Sequence

from vllm.inputs import EmbedsPrompt, TokensPrompt

from dynamo.vllm.multimodal_utils.custom_encoder.backend import ArtifactT


class CustomEncoderAdapter(ABC, Generic[ArtifactT]):
    """Translate encoder artifacts for one resolved downstream decoder."""

    @abstractmethod
    def prepare_prompt(
        self,
        token_ids: list[int],
        artifacts: Sequence[ArtifactT],
    ) -> EmbedsPrompt | TokensPrompt:
        """Validate encoder artifacts and build the final vLLM prompt.

        Runs per request, on the request's own coroutine — unlike
        ``VisionEncoderBackend.forward_batch``, a failure here is scoped to the
        one caller.

        Args:
            token_ids: Tokenized prompt containing the image placeholders.
            artifacts: Opaque values returned by the encoder backend, in image
                order. Each adapter defines and validates its concrete artifact
                contract.

        Raises:
            ValueError: the artifacts do not satisfy this adapter's contract, or
                do not line up with the placeholders in ``token_ids`` — a
                request-level fault the caller can act on. Dynamo maps this to
                ``Backend(InvalidArgument)``, answering **HTTP 400 with the
                message forwarded to the client verbatim**, so write it for that
                audience: state the mismatch in terms of the request and leave
                file paths, tracebacks, tensor dumps and weight identifiers out.
                Anything the caller cannot act on belongs in a log line.
            TypeError: an artifact is the wrong type entirely. Treated
                identically to ``ValueError``; the same message rules apply.

        Any other exception type is read as an engine fault rather than a bad
        request: the caller gets a sanitized 5xx and the message stays in the
        server log. Raise those freely — and prefer them whenever the request is
        not actually at fault, since a 400 tells the caller not to retry.
        """
