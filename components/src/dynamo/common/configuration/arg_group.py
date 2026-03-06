# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base ArgGroup interface."""
import argparse
from abc import ABC, abstractmethod


class ArgGroup(ABC):
    """
    Base interface for configuration groups.

    Each ArgGroup represents a domain of configuration parameters with clear ownership.

    Examples:
        >>> import argparse
        >>> from dynamo.common.configuration.arg_group import ArgGroup
        >>> from dynamo.common.configuration.utils import add_argument
        >>>
        >>> class MyArgGroup(ArgGroup):
        ...     def add_arguments(self, parser) -> None:
        ...         add_argument(
        ...             parser,
        ...             flag_name="--my-option",
        ...             env_var="DYN_MY_OPTION",
        ...             default="value",
        ...             help="A custom option.",
        ...         )
        >>>
        >>> parser = argparse.ArgumentParser()
        >>> MyArgGroup().add_arguments(parser)
    """

    @abstractmethod
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """
        Register CLI arguments owned by this group.

        This method must be side-effect free beyond parser mutation.
        It must not depend on runtime state or other groups.

        Args:
            parser: argparse.ArgumentParser or argument group
        """
        ...
