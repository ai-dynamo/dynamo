# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base ArgGroup interface."""
from abc import ABC, abstractmethod
from dataclasses import fields
from typing import Any, Mapping, Optional, Type


class ArgGroup(ABC):
    """
    Base interface for all configuration groups.

    Each ArgGroup represents a domain of configuration parameters with clear ownership.
    Groups are composed into a ConfigurationBuilder based on component requirements.
    """

    #: Canonical domain name (e.g., "dynamo-runtime", "dynamo-vllm")
    name: str

    #: If set, resolve() is implemented by filtering raw to this dataclass's fields and constructing it.
    #: Subclasses with custom resolve logic leave this None.
    config_class: Optional[Type[Any]] = None

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

    def resolve(self, raw: Mapping[str, Any]) -> Any:
        """
        Normalize and structurally validate raw inputs.
        Produce a domain-scoped configuration object.

        Must not perform semantic interpretation or cross-group logic.
        Default implementation: if config_class is set, build it from raw using only its dataclass fields.
        Override for custom logic (e.g. parsing lists, renames).

        Args:
            raw: Raw parsed arguments as dict

        Returns:
            Domain-specific config object (typically a dataclass)
        """
        cls = self.config_class
        if cls is None:
            raise NotImplementedError(
                f"{type(self).__name__} must set config_class or override resolve()"
            )
        field_names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in field_names})

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
