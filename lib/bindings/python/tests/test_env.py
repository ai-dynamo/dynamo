# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import warnings
from unittest.mock import patch

from dynamo.runtime.env import get_env, _to_bool


class TestToBool:
    """Test the _to_bool function."""

    def test_none_returns_false(self):
        assert _to_bool(None) is False

    def test_empty_string_returns_false(self):
        assert _to_bool("") is False

    def test_whitespace_only_returns_false(self):
        assert _to_bool("   ") is False

    def test_truthy_values(self):
        truthy_values = ["1", "true", "t", "yes", "y", "on"]
        for val in truthy_values:
            assert _to_bool(val) is True, f"'{val}' should be True"

    def test_truthy_values_case_insensitive(self):
        truthy_values = ["TRUE", "True", "T", "YES", "Yes", "ON", "On"]
        for val in truthy_values:
            assert _to_bool(val) is True, f"'{val}' should be True"

    def test_truthy_values_with_whitespace(self):
        truthy_values = [" 1 ", "  true  ", "\ttrue\t", "\nyes\n"]
        for val in truthy_values:
            assert _to_bool(val) is True, f"'{val}' should be True"

    def test_falsy_values(self):
        falsy_values = ["0", "false", "f", "no", "n", "off"]
        for val in falsy_values:
            assert _to_bool(val) is False, f"'{val}' should be False"

    def test_falsy_values_case_insensitive(self):
        falsy_values = ["FALSE", "False", "F", "NO", "No", "OFF", "Off"]
        for val in falsy_values:
            assert _to_bool(val) is False, f"'{val}' should be False"

    def test_falsy_values_with_whitespace(self):
        falsy_values = [" 0 ", "  false  ", "\tfalse\t", "\nno\n"]
        for val in falsy_values:
            assert _to_bool(val) is False, f"'{val}' should be False"

    def test_invalid_values_raise_error(self):
        invalid_values = ["invalid", "2", "maybe", "unknown", "true1", "false0"]
        for val in invalid_values:
            with pytest.raises(ValueError, match=f"Cannot interpret '{val}' as boolean"):
                _to_bool(val)


class TestGetEnv:
    """Test the get_env function."""

    def test_new_env_var_exists(self):
        with patch.dict('os.environ', {'NEW_VAR': 'new_value'}):
            result = get_env('NEW_VAR', 'OLD_VAR', 'default')
            assert result == 'new_value'

    def test_old_env_var_exists_with_warning(self, caplog):
        with patch.dict('os.environ', {'OLD_VAR': 'old_value'}, clear=True):
            result = get_env('NEW_VAR', 'OLD_VAR', 'default')
            assert result == 'old_value'
            assert len(caplog.records) == 1
            assert "DeprecationWarning" in caplog.records[0].message
            assert "deprecated" in caplog.records[0].message
            assert "NEW_VAR" in caplog.records[0].message
            assert "OLD_VAR" in caplog.records[0].message

    def test_both_env_vars_exist_prefers_new(self, caplog):
        with patch.dict('os.environ', {'NEW_VAR': 'new_value', 'OLD_VAR': 'old_value'}):
            result = get_env('NEW_VAR', 'OLD_VAR', 'default')
            assert result == 'new_value'
            assert len(caplog.records) == 0  # No warning when new var exists

    def test_neither_env_var_exists_returns_default(self):
        with patch.dict('os.environ', {}, clear=True):
            result = get_env('NEW_VAR', 'OLD_VAR', 'default_value')
            assert result == 'default_value'

    def test_neither_env_var_exists_no_default(self):
        with patch.dict('os.environ', {}, clear=True):
            result = get_env('NEW_VAR', 'OLD_VAR')
            assert result is None

    def test_no_old_var_specified(self):
        with patch.dict('os.environ', {}, clear=True):
            result = get_env('NEW_VAR', default='default_value')
            assert result == 'default_value'

    def test_string_mode_returns_string(self):
        with patch.dict('os.environ', {'NEW_VAR': 'true'}):
            result = get_env('NEW_VAR', as_bool=False)
            assert result == 'true'
            assert isinstance(result, str)

    def test_bool_mode_new_var_truthy(self):
        with patch.dict('os.environ', {'NEW_VAR': 'true'}):
            result = get_env('NEW_VAR', as_bool=True)
            assert result is True
            assert isinstance(result, bool)

    def test_bool_mode_new_var_falsy(self):
        with patch.dict('os.environ', {'NEW_VAR': 'false'}):
            result = get_env('NEW_VAR', as_bool=True)
            assert result is False
            assert isinstance(result, bool)

    def test_bool_mode_old_var_with_warning(self, caplog):
        with patch.dict('os.environ', {'OLD_VAR': 'yes'}, clear=True):
            result = get_env('NEW_VAR', 'OLD_VAR', as_bool=True)
            assert result is True
            assert isinstance(result, bool)
            assert len(caplog.records) == 1
            assert "DeprecationWarning" in caplog.records[0].message
            assert "deprecated" in caplog.records[0].message

    def test_bool_mode_default_string(self):
        with patch.dict('os.environ', {}, clear=True):
            result = get_env('NEW_VAR', default='true', as_bool=True)
            assert result is True
            assert isinstance(result, bool)

    def test_bool_mode_default_none(self):
        with patch.dict('os.environ', {}, clear=True):
            result = get_env('NEW_VAR', as_bool=True)
            assert result is False
            assert isinstance(result, bool)

    def test_bool_mode_empty_string(self):
        with patch.dict('os.environ', {'NEW_VAR': ''}):
            result = get_env('NEW_VAR', as_bool=True)
            assert result is False
            assert isinstance(result, bool)

    def test_bool_mode_invalid_value_raises_error(self):
        with patch.dict('os.environ', {'NEW_VAR': 'invalid'}):
            with pytest.raises(ValueError, match="Cannot interpret 'invalid' as boolean"):
                get_env('NEW_VAR', as_bool=True)

    def test_warning_stacklevel(self, caplog):
        """Test that deprecation warning is logged."""
        with patch.dict('os.environ', {'OLD_VAR': 'value'}, clear=True):
            get_env('NEW_VAR', 'OLD_VAR')
            assert len(caplog.records) == 1
            # The warning should contain the deprecation message
            assert "DeprecationWarning" in caplog.records[0].message
            assert "deprecated" in caplog.records[0].message
