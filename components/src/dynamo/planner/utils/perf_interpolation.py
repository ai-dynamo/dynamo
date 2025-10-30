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


import json
import logging
import os
from typing import Optional

import numpy as np
import scipy

from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


MISSING_PROFILING_DATA_ERROR_MESSAGE = (
    "SLA-Planner requires pre-deployment profiling results to run.\n"
    "Please follow /docs/benchmarks/sla_driven_profiling.md to run the profiling first,\n"
    "and make sure the profiling results are present in --profile-results-dir."
)


class PrefillInterpolator:
    """
    Takes input from results of pre-deployment performance profiling to interpolate
    throughput/gpu and TTFT for a given ISL.
    """

    def __init__(
        self,
        profile_results_dir: Optional[str] = None,
        raw_data: Optional[dict] = None,
    ):
        if profile_results_dir:
            prefill_npz_fn = (
                f"{profile_results_dir}/selected_prefill_interpolation/raw_data.npz"
            )
            try:
                with np.load(prefill_npz_fn) as raw_data:
                    self.prefill_isl = raw_data["prefill_isl"]
                    self.prefill_ttft = raw_data["prefill_ttft"]  # in milliseconds
                    self.prefill_thpt_per_gpu = raw_data["prefill_thpt_per_gpu"]
            except FileNotFoundError:
                # Fallback to JSON provided via ConfigMap mounted at profile_results_dir
                json_fn = os.path.join(profile_results_dir, "prefill_raw_data.json")
                try:
                    with open(json_fn, "r") as f:
                        data = json.load(f)
                        self.prefill_isl = np.array(data["prefill_isl"])  # type: ignore[index]
                        self.prefill_ttft = np.array(data["prefill_ttft"])  # type: ignore[index]
                        self.prefill_thpt_per_gpu = np.array(data["prefill_thpt_per_gpu"])  # type: ignore[index]
                except FileNotFoundError:
                    logger.error(
                        f"Prefill interpolation files not found: {prefill_npz_fn} and {json_fn}\n"
                        f"{MISSING_PROFILING_DATA_ERROR_MESSAGE}"
                    )
                    exit(1)

        elif raw_data:
            self.prefill_isl = raw_data["prefill_isl"]
            self.prefill_ttft = raw_data["prefill_ttft"]  # in milliseconds
            self.prefill_thpt_per_gpu = raw_data["prefill_thpt_per_gpu"]
        else:
            raise ValueError("Either profile_results_dir or raw_data must be provided")

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

    def __init__(
        self,
        profile_results_dir: Optional[str] = None,
        resolution: int = 100,
        raw_data: Optional[dict] = None,
    ):
        if profile_results_dir:
            decode_npz_fn = (
                f"{profile_results_dir}/selected_decode_interpolation/raw_data.npz"
            )
            try:
                with np.load(decode_npz_fn) as raw_data:
                    self.x_kv_usage = raw_data["x_kv_usage"]
                    self.y_context_length = raw_data["y_context_length"]
                    self.z_itl = raw_data["z_itl"]
                    self.z_thpt_per_gpu = raw_data["z_thpt_per_gpu"]
                    self.max_kv_tokens = raw_data["max_kv_tokens"][0]
            except FileNotFoundError:
                # Fallback to JSON provided via ConfigMap mounted at profile_results_dir
                json_fn = os.path.join(profile_results_dir, "decode_raw_data.json")
                try:
                    with open(json_fn, "r") as f:
                        data = json.load(f)
                        self.x_kv_usage = np.array(data["x_kv_usage"])  # type: ignore[index]
                        self.y_context_length = np.array(data["y_context_length"])  # type: ignore[index]
                        self.z_itl = np.array(data["z_itl"])  # type: ignore[index]
                        self.z_thpt_per_gpu = np.array(data["z_thpt_per_gpu"])  # type: ignore[index]
                        self.max_kv_tokens = int(data["max_kv_tokens"])  # type: ignore[index]
                except FileNotFoundError:
                    logger.error(
                        f"Decode interpolation files not found: {decode_npz_fn} and {json_fn}\n"
                        f"{MISSING_PROFILING_DATA_ERROR_MESSAGE}"
                    )
                    exit(1)
        elif raw_data:
            self.x_kv_usage = raw_data["x_kv_usage"]
            self.y_context_length = raw_data["y_context_length"]
            self.z_itl = raw_data["z_itl"]
            self.z_thpt_per_gpu = raw_data["z_thpt_per_gpu"]
            self.max_kv_tokens = raw_data["max_kv_tokens"][0]
        else:
            raise ValueError("Either profile_results_dir or raw_data must be provided")

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
        # ITL values are in milliseconds

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

    def find_best_throughput_per_gpu(
        self, itl: float, context_length: float
    ) -> tuple[float, float, float]:
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
                return (
                    self.thpt_interpolator[iy, ix],
                    self.itl_interpolator[iy, ix],
                    self.xi[ix],
                )
        return self.thpt_interpolator[iy, 0], self.itl_interpolator[iy, 0], self.xi[0]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_results_dir", type=str, required=True)
    parser.add_argument("--isl", type=int, default=3000)
    parser.add_argument("--osl", type=int, default=150)
    parser.add_argument("--ttft", type=float, default=100.0, help="in milliseconds")
    parser.add_argument("--itl", type=float, default=10.0, help="in milliseconds")
    args = parser.parse_args()

    print(f"ISL={args.isl}, OSL={args.osl}")
    print(f"TTFT={args.ttft}ms, ITL={args.itl}ms")
    print(f"Using profile results from {args.profile_results_dir}")
    print("")

    # first interpolate prefill
    print("Interpolating prefill performance ...")
    prefill_interpolator = PrefillInterpolator(args.profile_results_dir)

    est_ttft = prefill_interpolator.interpolate_ttft(args.isl)
    est_thpt_per_gpu = prefill_interpolator.interpolate_thpt_per_gpu(args.isl)

    if est_ttft <= args.ttft:
        print(
            f"\tEstimated TTFT={est_ttft:.2f}ms <= target TTFT={args.ttft:.2f}ms. Requests can queue {args.ttft - est_ttft:.2f}ms maximally while meeting TTFT SLA."
        )
    else:
        print(
            f"\tEstimated TTFT={est_ttft:.2f}ms > target TTFT={args.ttft:.2f}ms. Cannot meet TTFT SLA."
        )

    print(
        f"\tEstimated throughput: {est_thpt_per_gpu:.2f} tokens/s/gpu. Request rate at {est_thpt_per_gpu / args.isl:.2f} requests/s will saturate one GPU."
    )

    print("")

    # then interpolate decode
    decode_interpolator = DecodeInterpolator(args.profile_results_dir)

    print("Interpolating decode performance ...")
    context_length = args.isl + args.osl // 2
    print(f"\tAverage context length: isl + osl/2 = {context_length}.")
    (
        est_thpt_per_gpu,
        est_itl,
        est_kv_usage,
    ) = decode_interpolator.find_best_throughput_per_gpu(args.itl, context_length)
    if est_itl <= args.itl:
        print(
            f"\tEstimated ITL={est_itl:.2f}ms <= target ITL={args.itl:.2f}ms at {est_kv_usage*100:.2f}% active kv usage."
        )
        print(
            f"\tEstimated throughput: {est_thpt_per_gpu:.2f} token/s/gpu. Request rate at {est_thpt_per_gpu / args.osl:.2f} requests/s will saturate one GPU."
        )
    else:
        print(
            f"\tEstimated ITL={est_itl:.2f}ms > target ITL={args.itl:.2f}ms. Cannot meet ITL SLA."
        )
