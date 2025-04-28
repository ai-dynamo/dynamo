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


from dynamo.runtime.logging import (
    configure_dynamo_logging as configure_dynamo_runtime_logging,
)


def configure_dynamo_logging(
    service_name: str | None = None, worker_id: int | None = None
):
    """
    Pass through to the runtime logging configuration.
    """
    configure_dynamo_runtime_logging(service_name, worker_id)
