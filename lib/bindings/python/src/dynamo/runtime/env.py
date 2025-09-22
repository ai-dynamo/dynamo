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

import logging
import os

_TRUTHY = {"1", "true", "t", "yes", "y", "on"}
_FALSEY = {"0", "false", "f", "no", "n", "off"}


def get_env(
    name_new: str,
    name_old: str | None = None,
    default: str | None = None,
    as_bool: bool = False,
) -> str | bool | None:
    if (val := os.getenv(name_new)) is not None:
        return _to_bool(val) if as_bool else val
    if name_old is not None and (val := os.getenv(name_old)) is not None:
        logging.warning(
            f"DeprecationWarning: Environment variable '{name_old}' is deprecated, use '{name_new}' instead.",
            stacklevel=2,
        )
        return _to_bool(val) if as_bool else val
    return _to_bool(default) if as_bool else default


def _to_bool(val: str | None) -> bool:
    if val is None:
        return False
    s = val.strip().lower()
    if s == "" or s in _FALSEY:
        return False
    if s in _TRUTHY:
        return True
    raise ValueError(f"Cannot interpret '{val}' as boolean")
