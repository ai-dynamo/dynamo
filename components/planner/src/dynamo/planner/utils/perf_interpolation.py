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

import numpy as np
import scipy


class PrefillInterpolator:
    """
    Takes input from results of pre-deployment performance profiling to interpolate
    throughput/gpu and TTFT for a given ISL.
    """

    def __init__(self, profile_results_dir: str):
        prefill_npz_fn = (
            f"{profile_results_dir}/selected_prefill_interpolation/raw_data.npz"
        )

        with np.load(prefill_npz_fn) as raw_data:
            self.prefill_isl = raw_data["prefill_isl"]
            self.prefill_ttft = raw_data["prefill_ttft"]
            self.prefill_thpt_per_gpu = raw_data["prefill_thpt_per_gpu"]

        self.min_isl = min(self.prefill_isl)
        self.max_isl = max(self.prefill_isl)

        # perform 1d interpolation
        self.ttft_interpolator = scipy.interpolate.interp1d(
            self.prefill_isl, self.prefill_ttft, kind="cubic"
        )
        self.thpt_interpolator = scipy.interpolate.interp1d(
            self.prefill_isl, self.prefill_thpt_per_gpu, kind="cubic"
        )

    def interpolate_ttft(self, isl: float) -> float:
        isl = max(self.min_isl, min(isl, self.max_isl))
        return self.ttft_interpolator(isl)

    def interpolate_thpt_per_gpu(self, isl: float) -> float:
        isl = max(self.min_isl, min(isl, self.max_isl))
        return self.thpt_interpolator(isl)


class DecodeInterpolator:
    """
    Takes input from results of pre-deployment performance profiling to interpolate
    throughput/gpu and ITL for a given decode context length.
    """

    def __init__(self, profile_results_dir: str, resolution: int = 100):
        decode_npz_fn = (
            f"{profile_results_dir}/selected_decode_interpolation/raw_data.npz"
        )

        with np.load(decode_npz_fn) as raw_data:
            self.x_kv_usage = raw_data["x_kv_usage"]
            self.y_context_length = raw_data["y_context_length"]
            self.z_itl = raw_data["z_itl"]
            self.z_thpt_per_gpu = raw_data["z_thpt_per_gpu"]
            self.max_kv_tokens = raw_data["max_kv_tokens"][0]

        # pre-compute the interpolation grid for fast lookup
        self.resolution = resolution
        self.xi = np.linspace(0, 1, resolution)
        self.yi = np.linspace(0, max(self.y_context_length), resolution)
        self.X, self.Y = np.meshgrid(self.xi, self.yi)

        # perform 2d interpolation with fallback for NaN values
        self.itl_interpolator = scipy.interpolate.griddata(
            (self.x_kv_usage, self.y_context_length),
            self.z_itl,
            (self.X, self.Y),
            method="cubic",
        )
        # Fill NaN values using nearest neighbor interpolation
        nan_mask = np.isnan(self.itl_interpolator)
        if np.any(nan_mask):
            itl_nearest = scipy.interpolate.griddata(
                (self.x_kv_usage, self.y_context_length),
                self.z_itl,
                (self.X, self.Y),
                method="nearest",
            )
            self.itl_interpolator[nan_mask] = itl_nearest[nan_mask]
        self.itl_interpolator /= 1000  # convert ms to s

        self.thpt_interpolator = scipy.interpolate.griddata(
            (self.x_kv_usage, self.y_context_length),
            self.z_thpt_per_gpu,
            (self.X, self.Y),
            method="cubic",
        )
        # Fill NaN values using nearest neighbor interpolation
        nan_mask = np.isnan(self.thpt_interpolator)
        if np.any(nan_mask):
            thpt_nearest = scipy.interpolate.griddata(
                (self.x_kv_usage, self.y_context_length),
                self.z_thpt_per_gpu,
                (self.X, self.Y),
                method="nearest",
            )
            self.thpt_interpolator[nan_mask] = thpt_nearest[nan_mask]

    def compute_idx(self, concurrency: float, context_length: float) -> tuple[int, int]:
        kv_usage = concurrency * context_length / self.max_kv_tokens
        # Calculate x index (kv_usage)
        ix = int(
            np.clip(
                np.round((kv_usage - self.xi[0]) / (self.xi[1] - self.xi[0])),
                0,
                self.resolution - 1,
            )
        )
        # Calculate y index (context_length)
        iy = int(
            np.clip(
                np.round((context_length - self.yi[0]) / (self.yi[1] - self.yi[0])),
                0,
                self.resolution - 1,
            )
        )
        return ix, iy

    def interpolate_itl(self, concurrency: float, context_length: float) -> float:
        ix, iy = self.compute_idx(concurrency, context_length)
        return self.itl_interpolator[iy, ix]

    def interpolate_thpt_per_gpu(
        self, concurrency: float, context_length: float
    ) -> float:
        ix, iy = self.compute_idx(concurrency, context_length)
        return self.thpt_interpolator[iy, ix]

    def find_best_throughput_per_gpu(self, itl: float, context_length: float) -> float:
        # find the max kv_load that has itl <= target itl
        # here we cannot use binary search as interpolated itl might not be monotonic
        iy = int(
            np.clip(
                np.round((context_length - self.yi[0]) / (self.yi[1] - self.yi[0])),
                0,
                self.resolution - 1,
            )
        )
        iy = max(0, min(iy, self.resolution - 1))

        for ix in range(self.resolution - 1, -1, -1):
            if self.itl_interpolator[iy, ix] <= itl:
                return self.thpt_interpolator[iy, ix]
        return self.thpt_interpolator[iy, 0]
