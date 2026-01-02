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

from typing import Dict, List

import numpy as np
import pandas as pd
from tabulate import tabulate


def calculate_and_print_statistics(metrics: Dict[str, List[float]]) -> pd.DataFrame:
    """
    Calculate statistics for a dictionary of metrics and print them in a tabular format.

    Args:
        metrics: Dictionary where keys are metric names and values are lists of metric values

    Returns:
        pandas.DataFrame: DataFrame containing the calculated statistics
    """
    metric_names = []
    stats_data = []

    # Calculate statistics for each metric
    for metric_name, values in metrics.items():
        metric_names.append(metric_name)
        stats_data.append(
            {
                "Mean": np.mean(values),
                "Std Dev": np.std(values),
                "Min": np.min(values),
                "P25": np.percentile(values, 25),
                "Median": np.median(values),
                "P75": np.percentile(values, 75),
                "Max": np.max(values),
            }
        )

    # Replace the printing code with tabulate
    stats_df = pd.DataFrame(stats_data, index=metric_names)
    print(tabulate(stats_df, headers="keys", tablefmt="pretty", floatfmt=".2f"), "\n")

    return stats_df
