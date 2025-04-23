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

from dunamai import Style, Version

VERSION_FORMAT_AT_TAG = "{base}"
VERSION_FORMAT_AT_TAG_DIRTY = "{base}+d{timestamp}"
VERSION_FORMAT_POST_TAG = "{base}+post{distance}.{commit}"
VERSION_FORMAT_POST_TAG_DIRTY = "{base}+post{distance}.{commit}.d{timestamp}"

def calculate_version() -> str:
    version_info = Version.from_git()
    if version_info.distance == 0:
        if version_info.dirty:
            return version_info.serialize(format=VERSION_FORMAT_AT_TAG_DIRTY)
        else:
            return version_info.serialize(format=VERSION_FORMAT_AT_TAG)
    else:
        if version_info.dirty:
            return version_info.serialize(format=VERSION_FORMAT_POST_TAG_DIRTY)
        else:
            return version_info.serialize(format=VERSION_FORMAT_POST_TAG)

if __name__ == "__main__":
    print(calculate_version())