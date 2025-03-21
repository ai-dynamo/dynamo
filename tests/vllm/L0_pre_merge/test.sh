#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


set -xe

TEST_NAME="$(basename "$(dirname "$(readlink -f "$0")")")"
OUTPUT_DIR=${OUTPUT_DIR:-$PWD/.tests_outputs/${TEST_NAME}}
DYNAMO_REPO_DIR=${DYNAMO_REPO_DIR:-$PWD}

# TODO: recover mypy checks

pytest \
    --md-report \
    --md-report-verbose=6 \
    --md-report-output="${OUTPUT_DIR}/standard_pytest_report.md" \
    --junitxml="${OUTPUT_DIR}/standard_pytest_report.xml" \
    -m 'pre_merge' \
    $DYNAMO_REPO_DIR