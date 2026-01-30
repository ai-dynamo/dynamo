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

import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.random import Generator

logger = logging.getLogger(__name__)


def get_cdf(weights: List[float]) -> np.ndarray:
    cumsum = np.cumsum(weights)
    return cumsum / cumsum[-1]


def data_to_cdf(data: np.ndarray) -> Tuple[List[Any], np.ndarray]:
    sorted_counter: Dict[Any, int] = dict(sorted(Counter(data).items()))
    data_unique: List[Any] = list(sorted_counter.keys())
    counter_cdf: np.ndarray = get_cdf(list(sorted_counter.values()))
    return data_unique, counter_cdf


def sample_from_cdf(
    data: List[Any], cdf: np.ndarray, rng: Optional[Generator] = None
) -> Any:
    # NOTE: assumes (but does not verify) that the CDF is valid
    # CDF stands for cumulative distribution function
    assert len(data) == len(cdf)
    if rng is not None:
        return data[np.searchsorted(cdf, rng.random())]
    else:
        return data[np.searchsorted(cdf, np.random.rand())]


class EmpiricalSampler:
    """
    Takes data, learns from the pure empirical distribution, and samples directly from it.

    Args:
        data (Union[List[Any], np.ndarray]): The input data to learn the distribution from.
    """

    def __init__(self, data: Union[List[Any], np.ndarray]) -> None:
        self.rng = np.random.default_rng(0)
        self.empty_data = len(data) == 0
        if self.empty_data:
            logger.warning("Empty data provided to EmpiricalSampler")
        else:
            self.data, self.cdf = data_to_cdf(np.array(data))

    def sample(self) -> Any:
        if self.empty_data:
            return 0
        return sample_from_cdf(self.data, self.cdf, self.rng)
