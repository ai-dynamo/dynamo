# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ConfigurationBuilder for assembling configuration from ArgGroups."""

import argparse
import logging
from typing import Any, Dict, List

from .arg_group import ArgGroup

logger = logging.getLogger(__name__)


class ConfigurationBuilder:
    """
    Assembles argument parser from requested ArgGroups.

    The registry dynamically builds a parser based on which ArgGroups
    a component declares it needs, ensuring clear domain ownership.
    """

    def __init__(self, groups: List[ArgGroup]):
        """
        Initialize registry with required groups.

        Args:
            groups: List of ArgGroup instances to include
        """
        self.groups = groups
        self._check_duplicates()

    def _check_duplicates(self) -> None:
        """Verify no group name is duplicated."""
        seen = {}
        for group in self.groups:
            if group.name in seen:
                raise ValueError(f"Duplicate group name: {group.name}")
            seen[group.name] = group

    def build_parser(self, description: str = "") -> argparse.ArgumentParser:
        """
        Build argument parser from all included groups.

        Args:
            description: Parser description

        Returns:
            Configured ArgumentParser
        """
        parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.RawTextHelpFormatter,
        )

        # Let each group add its arguments
        for group in self.groups:
            logger.debug(f"Adding arguments from {group.name}")
            group.add_arguments(parser)

        return parser

    def parse_and_resolve(self, args=None) -> Dict[str, Any]:
        """
        Parse CLI args and resolve to typed config objects.

        Args:
            args: Command-line arguments (defaults to sys.argv)

        Returns:
            Dict mapping group name to resolved config object
        """
        parser = self.build_parser()

        # Parse arguments (including unknown for passthrough)
        parsed, unknown = parser.parse_known_args(args)
        raw_dict = vars(parsed)

        # Log passthrough args if any
        if unknown:
            logger.info(f"Passthrough arguments: {unknown}")
            raw_dict["_passthrough"] = unknown

        # Resolve each group
        configs = {}
        for group in self.groups:
            logger.debug(f"Resolving {group.name}")
            resolved = group.resolve(raw_dict)

            # Validate
            try:
                group.validate(resolved)
            except Exception as e:
                raise ValueError(f"Validation failed for {group.name}: {e}") from e

            # Use snake_case for dict keys
            key = group.name.replace("-", "_")
            configs[key] = resolved

        # Include passthrough arguments if any
        if "_passthrough" in raw_dict:
            configs["_passthrough"] = raw_dict["_passthrough"]

        return configs
