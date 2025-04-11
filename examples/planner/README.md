<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Planner

The planner is a component that monitors the state of the system and makes adjustments to the number of workers to ensure that the system is running efficiently.

## Usage
After you've deployed a dynamo graph - you can start the planner with the following command:
```bash
python components/planner.py --namespace <namespace>
```

## Backends
We currently support two backends:
1. `local` - uses dynamo serve to start/stop a system
2. `kuberentes` - uses the kuberentes api to adjust replicas of each component's resource definition