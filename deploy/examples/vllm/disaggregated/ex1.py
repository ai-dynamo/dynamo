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

from disaggregated.frontend import Frontend
from disaggregated.kv_router import Router
from disaggregated.processor import Processor
from disaggregated.worker import VllmWorker

# example 2 and 3: kv aware routing + worker
# kv.yaml
Frontend.link(Processor).link(Router).link(VllmWorker)

# example 4 and 5: only disag - issue with endpoint (probably because of routerless)
# disag.yaml
# Frontend.link(VllmWorker).link(PrefillWorker)

# example 6: disag with kv
# kv_with_disag.yaml
# Frontend.link(Processor).link(Router).link(VllmWorker).link(PrefillWorker)
