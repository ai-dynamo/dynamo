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

from abc import ABC, abstractmethod


# TODO: add ability to scale component to X replicas
class PlannerConnector(ABC):
    @abstractmethod
    async def get_component_replicas(self, component_name):
        """Get current number of replicas for a component"""
        pass

    @abstractmethod
    async def add_component(self, component_name):
        """Add a component to the planner"""
        pass

    @abstractmethod
    async def remove_component(self, component_name):
        """Remove a component from the planner"""
        pass

    @abstractmethod
    async def get_resource_usage(self, component_name):
        """Get resource usage for a component"""
        pass

    @abstractmethod
    async def get_system_topology(self):
        """Get system topology (components and their relationships)"""
        pass
