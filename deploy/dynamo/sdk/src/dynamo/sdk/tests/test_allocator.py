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

import unittest.mock as mock

import pytest

from dynamo.sdk.cli.allocator import NVIDIA_GPU, ResourceAllocator

pytestmark = pytest.mark.pre_merge


@pytest.fixture
def mock_system_resources():
    """Fixture to mock system resources with 4 GPUs and 8 CPUs."""
    with mock.patch("bentoml._internal.resource.system_resources") as mock_resources:
        mock_resources.return_value = {
            NVIDIA_GPU: ["0", "1", "2", "3"],
            "cpu": 8,
        }
        yield mock_resources


@pytest.fixture
def mock_service():
    """Fixture to create a mock BentoML service."""
    service_mock = mock.Mock()
    service_mock.name = "test_service"
    return service_mock


@pytest.fixture
def mock_services():
    """Fixture to mock the BentoML services configuration."""
    return {
        "test_service": {
            "resources": {"gpu": 1},
            "workers": 2,
        }
    }


class TestResourceAllocator:
    def test_init(self, mock_system_resources):
        """Test initialization of ResourceAllocator."""
        allocator = ResourceAllocator()

        assert allocator.remaining_gpus == 4
        assert allocator._available_gpus == [
            (1.0, 1.0),
            (1.0, 1.0),
            (1.0, 1.0),
            (1.0, 1.0),
        ]

    def test_assign_full_gpu(self, mock_system_resources):
        """Test assigning a full GPU."""
        allocator = ResourceAllocator()
        assigned = allocator.assign_gpus(1.0)

        assert assigned == [0]
        assert allocator.remaining_gpus == 3
        assert allocator._available_gpus[0] == (0.0, 1.0)

        # Assign another GPU
        assigned = allocator.assign_gpus(1.0)
        assert assigned == [1]
        assert allocator.remaining_gpus == 2

    def test_assign_fractional_gpu(self, mock_system_resources):
        """Test assigning a fractional GPU."""
        allocator = ResourceAllocator()

        # Assign 0.5 GPU
        assigned = allocator.assign_gpus(0.5)
        assert assigned == [0]
        assert allocator.remaining_gpus == 3
        assert allocator._available_gpus[0] == (0.5, 0.5)

        # Assign another 0.5 GPU (should take from the same GPU)
        assigned = allocator.assign_gpus(0.5)
        assert assigned == [0]
        assert allocator.remaining_gpus == 2
        assert allocator._available_gpus[0] == (0.0, 0.5)

    def test_assign_mixed_fractional_gpus(self, mock_system_resources):
        """Test assigning mixed fractions of GPUs."""
        allocator = ResourceAllocator()

        # Assign 0.3 GPU
        assigned = allocator.assign_gpus(0.3)
        assert assigned == [0]
        assert allocator._available_gpus[0] == (0.7, 0.3)

        # Assign 0.5 GPU (should go to a new GPU since 0.5 > remaining 0.3)
        assigned = allocator.assign_gpus(0.5)
        assert assigned == [1]
        assert allocator._available_gpus[1] == (0.5, 0.5)

    def test_assign_too_many_gpus(self, mock_system_resources):
        """Test assigning more GPUs than available with warning."""
        allocator = ResourceAllocator()

        with pytest.warns(ResourceWarning):
            assigned = allocator.assign_gpus(6.0)

        assert len(assigned) == 6
        assert allocator.remaining_gpus == 0

    def test_assign_float_gpu_larger_than_one(self, mock_system_resources):
        """Test that assigning a float GPU > 1 raises an exception."""
        allocator = ResourceAllocator()

        with pytest.raises(Exception):
            allocator.assign_gpus(1.5)
