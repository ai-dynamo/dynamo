# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inputs that control how one Sweeper Candidate becomes a DGD."""

from __future__ import annotations

import re
from dataclasses import dataclass

_RUNTIME_VERSION_PATTERN = re.compile(
    r"^(0|[1-9][0-9]{0,3})\.(0|[1-9][0-9]{0,3})\.(0|[1-9][0-9]{0,3})$"
)


def _runtime_version(runtime_image: str, override: str | None) -> str:
    if override is not None:
        return override.strip()

    image_without_digest = runtime_image.partition("@")[0]
    image_name = image_without_digest.rsplit("/", 1)[-1]
    _, separator, tag = image_name.rpartition(":")
    if separator and _RUNTIME_VERSION_PATTERN.fullmatch(tag):
        return tag
    raise ValueError(
        "runtime_image must have a canonical MAJOR.MINOR.PATCH tag when "
        "runtime_version_override is not set"
    )


@dataclass(frozen=True)
class DGDGenerationOptions:
    """Inputs that control how one Sweeper Candidate becomes a DGD."""

    runtime_image: str
    num_gpus_per_node: int
    runtime_version_override: str | None = None
    namespace: str | None = None

    def __post_init__(self) -> None:
        if not self.runtime_image.strip():
            raise ValueError("runtime_image must not be empty")
        if self.num_gpus_per_node < 1:
            raise ValueError("num_gpus_per_node must be positive")
        if (
            self.runtime_version_override is not None
            and not _RUNTIME_VERSION_PATTERN.fullmatch(
                self.runtime_version_override.strip()
            )
        ):
            raise ValueError(
                "runtime_version_override must be a canonical MAJOR.MINOR.PATCH version"
            )

        # AIC needs the Dynamo runtime version even when the DGD does not need an override.
        _runtime_version(self.runtime_image, self.runtime_version_override)

    @property
    def dynamo_runtime_version(self) -> str:
        """Return the Dynamo version declared by the override or image tag."""
        return _runtime_version(
            self.runtime_image,
            self.runtime_version_override,
        )


__all__ = ["DGDGenerationOptions"]
