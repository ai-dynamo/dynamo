# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base ArgGroup interface."""

from abc import ABC, abstractmethod
from typing import Any, Mapping


class ArgGroup(ABC):
    """
    Base interface for all configuration groups.

    Each ArgGroup represents a domain of configuration parameters with clear ownership.
    Groups are composed into a ConfigurationBuilder based on component requirements.
    """

    #: Canonical domain name (e.g., "dynamo-runtime", "dynamo-vllm")
    name: str

    @abstractmethod
    def add_arguments(self, parser) -> None:
        """
        Register CLI arguments owned by this group.

        This method must be side-effect free beyond parser mutation.
        It must not depend on runtime state or other groups.

        Args:
            parser: argparse.ArgumentParser or argument group
        """
        ...

    @abstractmethod
    def resolve(self, raw: Mapping[str, Any]) -> Any:
        """
        Normalize and structurally validate raw inputs.
        Produce a domain-scoped configuration object.

        Must not perform semantic interpretation or cross-group logic.

        Args:
            raw: Raw parsed arguments as dict

        Returns:
            Domain-specific config object (typically a dataclass)
        """
        ...

    def validate(self, resolved: Any) -> None:
        """
        Optional additional structural validation.
        Must not inspect other groups or runtime state.

        Args:
            resolved: The config object returned by resolve()

        Raises:
            ValueError: If validation fails
        """
        pass
