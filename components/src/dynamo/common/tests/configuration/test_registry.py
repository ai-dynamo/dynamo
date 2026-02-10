# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CompositeRegistry class."""

import argparse
from dataclasses import dataclass
from typing import Any, Mapping

import pytest

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.registry import CompositeRegistry


# Test ArgGroups
@dataclass
class Group1Config:
    """Config for group 1."""

    param1: str
    param2: int


class TestGroup1(ArgGroup):
    """Test group 1."""

    name = "group1"

    def add_arguments(self, parser):
        g = parser.add_argument_group("Group 1")
        g.add_argument("--param1", type=str, default="default1")
        g.add_argument("--param2", type=int, default=10)

    def resolve(self, raw: Mapping[str, Any]) -> Group1Config:
        return Group1Config(param1=raw["param1"], param2=raw["param2"])

    def validate(self, resolved: Group1Config) -> None:
        if resolved.param2 < 0:
            raise ValueError("param2 must be non-negative")


@dataclass
class Group2Config:
    """Config for group 2."""

    flag: bool


class TestGroup2(ArgGroup):
    """Test group 2."""

    name = "group2"

    def add_arguments(self, parser):
        g = parser.add_argument_group("Group 2")
        g.add_argument("--flag", action="store_true")

    def resolve(self, raw: Mapping[str, Any]) -> Group2Config:
        return Group2Config(flag=raw["flag"])


class TestCompositeRegistry:
    """Test CompositeRegistry class."""

    def test_init_single_group(self):
        """Test initialization with single group."""
        group = TestGroup1()
        registry = CompositeRegistry([group])

        assert len(registry.groups) == 1
        assert registry.groups[0].name == "group1"

    def test_init_multiple_groups(self):
        """Test initialization with multiple groups."""
        groups = [TestGroup1(), TestGroup2()]
        registry = CompositeRegistry(groups)

        assert len(registry.groups) == 2

    def test_init_empty_list(self):
        """Test initialization with empty list."""
        registry = CompositeRegistry([])

        assert len(registry.groups) == 0

    def test_duplicate_group_names_raises(self):
        """Test that duplicate group names raise error."""
        group1a = TestGroup1()
        group1b = TestGroup1()  # Same name

        with pytest.raises(ValueError, match="Duplicate group name"):
            CompositeRegistry([group1a, group1b])

    def test_build_parser_returns_parser(self):
        """Test that build_parser returns ArgumentParser."""
        registry = CompositeRegistry([TestGroup1()])

        parser = registry.build_parser()

        assert isinstance(parser, argparse.ArgumentParser)

    def test_build_parser_adds_arguments(self):
        """Test that build_parser adds arguments from groups."""
        registry = CompositeRegistry([TestGroup1()])

        parser = registry.build_parser()

        # Should be able to parse arguments from group
        args = parser.parse_args(["--param1", "test", "--param2", "42"])
        assert args.param1 == "test"
        assert args.param2 == 42

    def test_build_parser_combines_multiple_groups(self):
        """Test that build_parser combines arguments from multiple groups."""
        registry = CompositeRegistry([TestGroup1(), TestGroup2()])

        parser = registry.build_parser()

        # Should have arguments from both groups
        args = parser.parse_args(["--param1", "test", "--flag"])
        assert args.param1 == "test"
        assert args.flag is True

    def test_parse_and_resolve_single_group(self):
        """Test parse_and_resolve with single group."""
        registry = CompositeRegistry([TestGroup1()])

        configs = registry.parse_and_resolve(["--param1", "myvalue", "--param2", "99"])

        assert "group1" in configs
        config = configs["group1"]
        assert isinstance(config, Group1Config)
        assert config.param1 == "myvalue"
        assert config.param2 == 99

    def test_parse_and_resolve_multiple_groups(self):
        """Test parse_and_resolve with multiple groups."""
        registry = CompositeRegistry([TestGroup1(), TestGroup2()])

        configs = registry.parse_and_resolve(["--param1", "test", "--flag"])

        assert "group1" in configs
        assert "group2" in configs
        assert configs["group1"].param1 == "test"
        assert configs["group2"].flag is True

    def test_parse_and_resolve_uses_defaults(self):
        """Test that defaults are used when args not provided."""
        registry = CompositeRegistry([TestGroup1()])

        configs = registry.parse_and_resolve([])

        assert configs["group1"].param1 == "default1"
        assert configs["group1"].param2 == 10

    def test_parse_and_resolve_handles_unknown_args(self):
        """Test that unknown arguments are captured in passthrough."""
        registry = CompositeRegistry([TestGroup1()])

        configs = registry.parse_and_resolve(
            ["--param1", "test", "--unknown-flag", "value", "--another-unknown"]
        )

        assert "_passthrough" in configs
        assert "--unknown-flag" in configs["_passthrough"]
        assert "value" in configs["_passthrough"]
        assert "--another-unknown" in configs["_passthrough"]

    def test_parse_and_resolve_no_unknown_args(self):
        """Test passthrough not included when no unknown args."""
        registry = CompositeRegistry([TestGroup1()])

        configs = registry.parse_and_resolve(["--param1", "test"])

        # _passthrough should not be present if no unknown args
        assert "_passthrough" not in configs

    def test_group_name_with_hyphens_converts_to_underscores(self):
        """Test that group names with hyphens convert to underscores in keys."""

        class HyphenatedGroup(ArgGroup):
            name = "my-hyphenated-group"

            def add_arguments(self, parser):
                parser.add_argument("--test")

            def resolve(self, raw):
                return {"test": raw.get("test")}

        registry = CompositeRegistry([HyphenatedGroup()])
        configs = registry.parse_and_resolve([])

        assert "my_hyphenated_group" in configs

    def test_validation_called_and_passes(self):
        """Test that validation is called and passes for valid config."""
        registry = CompositeRegistry([TestGroup1()])

        # Should not raise
        configs = registry.parse_and_resolve(["--param2", "10"])
        assert configs["group1"].param2 == 10

    def test_validation_called_and_fails(self):
        """Test that validation is called and fails for invalid config."""
        registry = CompositeRegistry([TestGroup1()])

        with pytest.raises(ValueError, match="Validation failed for group1"):
            registry.parse_and_resolve(["--param2", "-5"])

    def test_parse_and_resolve_with_description(self):
        """Test that description can be set."""
        registry = CompositeRegistry([TestGroup1()])

        parser = registry.build_parser("My test application")

        help_text = parser.format_help()
        assert "My test application" in help_text

    def test_empty_registry_parses_successfully(self):
        """Test that empty registry still works."""
        registry = CompositeRegistry([])

        configs = registry.parse_and_resolve([])

        assert isinstance(configs, dict)
        assert len(configs) == 0
