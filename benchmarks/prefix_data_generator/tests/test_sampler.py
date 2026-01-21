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

from collections import Counter

import numpy as np
from prefix_data_generator.sampler import EmpiricalSampler


def test_empirical_sampler_distribution():
    # Create a test array with equal numbers of 1, 2, and 3
    test_data = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])

    # Create the sampler
    sampler = EmpiricalSampler(test_data)

    # Sample 1000 times
    samples = [sampler.sample() for _ in range(1000)]

    # Count occurrences of each value
    counts = Counter(samples)

    # Verify each number (1, 2, 3) appears between 300 and 400 times
    for value in [1, 2, 3]:
        assert (
            300 <= counts[value] <= 400
        ), f"Value {value} appeared {counts[value]} times, expected 300-400 times"

    # Verify no other values appear in the samples
    assert set(counts.keys()) == {
        1,
        2,
        3,
    }, f"Unexpected values in samples: {set(counts.keys()) - {1, 2, 3}}"
