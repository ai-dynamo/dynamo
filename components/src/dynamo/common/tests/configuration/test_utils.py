# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for configuration utility functions."""
import argparse

import pytest

from dynamo.common.configuration.utils import (
    add_argument,
    add_negatable_bool_argument,
    env_or_default,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class TestEnvOrDefault:
    """Test env_or_default function."""

    def test_returns_default_when_env_not_set(self, monkeypatch):
        """Test returns default value when env var not set."""
        monkeypatch.delenv("TEST_VAR", raising=False)

        result = env_or_default("TEST_VAR", "default_value")
        assert result == "default_value"

    def test_returns_env_when_set(self, monkeypatch):
        """Test returns env value when set."""
        monkeypatch.setenv("TEST_VAR", "env_value")

        result = env_or_default("TEST_VAR", "default_value")
        assert result == "env_value"

    def test_bool_conversion_true(self, monkeypatch):
        """Test bool conversion for true values."""
        test_cases = ["true", "True", "1", "yes", "YES", "on", "ON"]

        for value in test_cases:
            monkeypatch.setenv("TEST_BOOL", value)
            result = env_or_default("TEST_BOOL", False)
            assert result is True, f"Failed for value: {value}"

    def test_bool_conversion_false(self, monkeypatch):
        """Test bool conversion for false values."""
        test_cases = ["false", "False", "0", "no", "NO", "off", "OFF"]

        for value in test_cases:
            monkeypatch.setenv("TEST_BOOL", value)
            result = env_or_default("TEST_BOOL", True)
            assert result is False, f"Failed for value: {value}"

    def test_int_conversion(self, monkeypatch):
        """Test int conversion."""
        monkeypatch.setenv("TEST_INT", "42")

        result = env_or_default("TEST_INT", 0)
        assert result == 42
        assert isinstance(result, int)

    def test_float_conversion(self, monkeypatch):
        """Test float conversion."""
        monkeypatch.setenv("TEST_FLOAT", "3.14")

        result = env_or_default("TEST_FLOAT", 0.0)
        assert result == 3.14
        assert isinstance(result, float)

    def test_string_passthrough(self, monkeypatch):
        """Test string values are passed through."""
        monkeypatch.setenv("TEST_STR", "hello world")

        result = env_or_default("TEST_STR", "default")
        assert result == "hello world"

    def test_preserves_default_type(self, monkeypatch):
        """Test that default type is preserved when env not set."""
        monkeypatch.delenv("TEST_VAR", raising=False)

        # String
        assert isinstance(env_or_default("TEST_VAR", "str"), str)
        # Int
        assert isinstance(env_or_default("TEST_VAR", 42), int)
        # Float
        assert isinstance(env_or_default("TEST_VAR", 3.14), float)
        # Bool
        assert isinstance(env_or_default("TEST_VAR", True), bool)

    def test_none_default_with_no_value_type_returns_raw_env_value(self, monkeypatch):
        """Test env value is returned as string when no type info is available."""
        monkeypatch.setenv("TEST_VAR", "env_value")

        result = env_or_default("TEST_VAR", None, value_type=None)
        assert result == "env_value"
        assert isinstance(result, str)


class TestAddArgument:
    """Test add_argument function."""

    def test_callable_type_with_none_default_uses_env_and_validates(self, monkeypatch):
        """Test callable arg_type works when default is None and env var is set."""
        monkeypatch.setenv("TEST_MODEL_NAME", "  model-A  ")
        parser = argparse.ArgumentParser()

        def validate_model_name(value: str) -> str:
            if len(value.strip()) == 0:
                raise argparse.ArgumentTypeError("model-name must be non-empty")
            return value.strip()

        add_argument(
            parser,
            flag_name="--model-name",
            env_var="TEST_MODEL_NAME",
            default=None,
            help="Model name",
            arg_type=validate_model_name,
        )

        args = parser.parse_args([])
        assert args.model_name == "model-A"

    def test_callable_type_with_none_default_uses_none_when_env_unset(
        self, monkeypatch
    ):
        """Test callable arg_type keeps None default when env var is not set."""
        monkeypatch.delenv("TEST_MODEL_NAME", raising=False)
        parser = argparse.ArgumentParser()

        def validate_model_name(value: str) -> str:
            if len(value.strip()) == 0:
                raise argparse.ArgumentTypeError("model-name must be non-empty")
            return value.strip()

        add_argument(
            parser,
            flag_name="--model-name",
            env_var="TEST_MODEL_NAME",
            default=None,
            help="Model name",
            arg_type=validate_model_name,
        )

        args = parser.parse_args([])
        assert args.model_name is None

    def test_callable_type_with_invalid_env_value_fails_parse(self, monkeypatch):
        """Test invalid env value still fails validation via argparse type callable."""
        monkeypatch.setenv("TEST_MODEL_NAME", "   ")
        parser = argparse.ArgumentParser()

        def validate_model_name(value: str) -> str:
            if len(value.strip()) == 0:
                raise argparse.ArgumentTypeError("model-name must be non-empty")
            return value.strip()

        add_argument(
            parser,
            flag_name="--model-name",
            env_var="TEST_MODEL_NAME",
            default=None,
            help="Model name",
            arg_type=validate_model_name,
        )

        with pytest.raises(SystemExit):
            parser.parse_args([])


class TestAddNegatableBool:
    """Test add_negatable_bool function."""

    def test_positive_flag(self):
        """Test that --flag added."""
        parser = argparse.ArgumentParser()
        add_negatable_bool_argument(
            parser,
            flag_name="--enable-feature",
            env_var="TEST_ENABLE",
            default=True,
            help="Enable feature",
        )

        # Test positive flag
        args = parser.parse_args(["--enable-feature"])
        assert args.enable_feature is True

    def test_negative_flag(self):
        """Test that --no-flag are added."""
        # Test negative flag
        parser = argparse.ArgumentParser()
        add_negatable_bool_argument(
            parser,
            flag_name="--enable-feature",
            env_var="TEST_ENABLE",
            default=True,
            help="Enable feature",
        )
        args = parser.parse_args(["--no-enable-feature"])
        assert args.enable_feature is False

    def test_uses_default_when_no_flag(self, monkeypatch):
        """Test uses default value when no flag provided."""
        monkeypatch.delenv("TEST_ENABLE", raising=False)

        parser = argparse.ArgumentParser()
        add_negatable_bool_argument(
            parser,
            flag_name="--enable-feature",
            env_var="TEST_ENABLE",
            default=True,
            help="Enable feature",
        )

        args = parser.parse_args([])
        assert args.enable_feature is True

    def test_uses_env_var_when_set(self, monkeypatch):
        """Test uses environment variable when set."""
        monkeypatch.setenv("TEST_ENABLE", "false")

        parser = argparse.ArgumentParser()
        add_negatable_bool_argument(
            parser,
            flag_name="--enable-feature",
            env_var="TEST_ENABLE",
            default=True,
            help="Enable feature",
        )

        args = parser.parse_args([])
        assert args.enable_feature is False

    def test_converts_hyphens_to_underscores(self):
        """Test that flag name with hyphens converts to underscores in dest."""
        parser = argparse.ArgumentParser()
        add_negatable_bool_argument(
            parser,
            flag_name="--my-cool-feature",
            env_var="TEST_FEATURE",
            default=False,
            help="Cool feature",
        )

        args = parser.parse_args([])
        # Should have my_cool_feature attribute
        assert hasattr(args, "my_cool_feature")

    def test_help_includes_env_var(self):
        """Test that help text includes environment variable name."""
        parser = argparse.ArgumentParser()
        add_negatable_bool_argument(
            parser,
            flag_name="--feature",
            env_var="MY_ENV_VAR",
            default=True,
            help="Test feature",
        )

        help_text = parser.format_help()
        assert "MY_ENV_VAR" in help_text
        assert "Test feature" in help_text

    def test_help_shows_current_default(self, monkeypatch):
        """Test that help shows the default value."""
        monkeypatch.setenv("TEST_VAR", "true")

        parser = argparse.ArgumentParser()
        add_negatable_bool_argument(
            parser,
            flag_name="--feature",
            env_var="TEST_VAR",
            default=False,
            help="Test",
        )

        help_text = parser.format_help()
        assert "False" in help_text or "false" in help_text
