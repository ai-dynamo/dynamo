# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base ArgGroup interface."""
from abc import ABC, abstractmethod


class ArgGroup(ABC):
    """
    Base interface for configuration groups.

    Each ArgGroup represents a domain of configuration parameters with clear ownership.
    """

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
