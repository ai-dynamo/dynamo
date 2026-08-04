# check=skip=InvalidDefaultArgInFrom
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Required so callers select an available, runtime-compatible published release.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG RUNTIME_USER=dynamo

USER root
COPY --chmod=644 requirements.txt /tmp/riva-requirements.txt
RUN uv pip install \
    --system \
    --no-cache \
    --requirement /tmp/riva-requirements.txt

COPY --chown=${RUNTIME_USER}:root riva_nim/ /opt/riva-cascaded-pipeline/riva_nim/
ENV PYTHONPATH=/opt/riva-cascaded-pipeline
USER ${RUNTIME_USER}
