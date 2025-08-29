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
import logging
from typing import Optional, Union
try:
    from dynamo.planner.db.worker_profile_client import WorkerProfileClient
except ImportError:
    # Try relative import if the package is not installed
    try:
        from ..db.worker_profile_client import WorkerProfileClient
    except ImportError:
        # Try direct import if running as script
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from db.worker_profile_client import WorkerProfileClient

logger = logging.getLogger(__name__)


class PrefillInterpolator:
    """
    Takes input from results of pre-deployment performance profiling to interpolate
    throughput/gpu and TTFT for a given ISL.
    """

    def __init__(self, profile_results_dir: Optional[str] = None, 
                 prefill_isl: Optional[Union[np.ndarray, list]] = None,
                 prefill_ttft: Optional[Union[np.ndarray, list]] = None,
                 prefill_thpt_per_gpu: Optional[Union[np.ndarray, list]] = None):
        """
        Initialize PrefillInterpolator with either file-based or data-based input.
        
        Args:
            profile_results_dir: Path to profile results directory (file-based)
            prefill_isl: Array or list of input sequence lengths (data-based)
            prefill_ttft: Array or list of time to first token values in ms (data-based)
            prefill_thpt_per_gpu: Array or list of throughput per GPU values (data-based)
        """
        if profile_results_dir is not None:
            # File-based initialization
            prefill_npz_fn = (
                f"{profile_results_dir}/selected_prefill_interpolation/raw_data.npz"
            )

            with np.load(prefill_npz_fn) as raw_data:
                self.prefill_isl = raw_data["prefill_isl"]
                self.prefill_ttft = raw_data["prefill_ttft"] / 1000  # convert ms to s
                self.prefill_thpt_per_gpu = raw_data["prefill_thpt_per_gpu"]
        elif (prefill_isl is not None and prefill_ttft is not None and 
              prefill_thpt_per_gpu is not None):
            # Data-based initialization
            self.prefill_isl = np.array(prefill_isl)
            self.prefill_ttft = np.array(prefill_ttft) / 1000  # convert ms to s
            self.prefill_thpt_per_gpu = np.array(prefill_thpt_per_gpu)
        else:
            raise ValueError("Either profile_results_dir or db query args must be provided")

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

    def __init__(self, profile_results_dir: Optional[str] = None, 
                 resolution: int = 100,
                 x_kv_usage: Optional[Union[np.ndarray, list]] = None,
                 y_context_length: Optional[Union[np.ndarray, list]] = None,
                 z_itl: Optional[Union[np.ndarray, list]] = None,
                 z_thpt_per_gpu: Optional[Union[np.ndarray, list]] = None,
                 max_kv_tokens: Optional[int] = None):
        """
        Initialize DecodeInterpolator with either file-based or data-based input.
        
        Args:
            profile_results_dir: Path to profile results directory (file-based)
            resolution: Resolution for interpolation grid
            x_kv_usage: Array or list of KV usage ratios (data-based)
            y_context_length: Array or list of context lengths (data-based)
            z_itl: Array or list of inter-token latency values in ms (data-based)
            z_thpt_per_gpu: Array or list of throughput per GPU values (data-based)
            max_kv_tokens: Maximum KV tokens value (data-based)
        """
        if profile_results_dir is not None:
            # File-based initialization
            decode_npz_fn = (
                f"{profile_results_dir}/selected_decode_interpolation/raw_data.npz"
            )

            with np.load(decode_npz_fn) as raw_data:
                self.x_kv_usage = raw_data["x_kv_usage"]
                self.y_context_length = raw_data["y_context_length"]
                self.z_itl = raw_data["z_itl"]
                self.z_thpt_per_gpu = raw_data["z_thpt_per_gpu"]
                self.max_kv_tokens = raw_data["max_kv_tokens"][0]
        elif (x_kv_usage is not None and y_context_length is not None and 
              z_itl is not None and z_thpt_per_gpu is not None and 
              max_kv_tokens is not None):
            # Data-based initialization
            self.x_kv_usage = np.array(x_kv_usage)
            self.y_context_length = np.array(y_context_length)
            self.z_itl = np.array(z_itl)
            self.z_thpt_per_gpu = np.array(z_thpt_per_gpu)
            self.max_kv_tokens = int(max_kv_tokens)
        else:
            raise ValueError("Either profile_results_dir or db query args must be provided")

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

        # Set min/max attributes for convenience
        self.min_kv_usage = min(self.x_kv_usage)
        self.max_kv_usage = max(self.x_kv_usage)
        self.min_context_length = min(self.y_context_length)
        self.max_context_length = max(self.y_context_length)

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


async def create_prefill_interpolator(
    profile_results_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    tp_size: Optional[int] = None,
    backend: Optional[str] = None,
    gpu_count: Optional[int] = None,
    node_count: Optional[int] = None,
    dp_size: Optional[int] = None,
    pp_size: Optional[int] = None,
    gpu_type: Optional[str] = None,
    max_context_length: Optional[int] = None
) -> PrefillInterpolator:
    """
    Create a PrefillInterpolator from either file-based or database-based data.
    
    Args:
        profile_results_dir: Path to profile results directory (file-based)
        model_name: Model name for database lookup
        tp_size: Tensor parallelism size
        backend: Backend name (e.g., "vllm", "trtllm", "sglang")
        gpu_count: Number of GPUs
        node_count: Number of nodes
        dp_size: Data parallelism size
        pp_size: Pipeline parallelism size
        gpu_type: GPU type (e.g., "h100", "a100")
        max_context_length: Maximum context length
        
    Returns:
        PrefillInterpolator instance
    """
    if profile_results_dir is not None:
        # File-based initialization
        return PrefillInterpolator(profile_results_dir=profile_results_dir)
    
    elif model_name is not None and tp_size is not None:
        # Database-based initialization
        try:
            # Initialize WorkerProfileClient
            profile_client = WorkerProfileClient()
            
            # Get the latest profile data for the specified configuration
            # Query with the latest uniq_profile_id for the given parameters
            query = f"""
            SELECT uniq_profile_id
            FROM {profile_client.table_name}
            WHERE model_name = '{model_name.replace("'", "''")}'
            AND tp = {tp_size}
            AND worker_type = 'prefill'
            """
            
            # Add optional filters
            if backend is not None:
                query += f" AND backend = '{backend.replace("'", "''")}'"
            if gpu_count is not None:
                query += f" AND gpu_count = {gpu_count}"
            if node_count is not None:
                query += f" AND node_count = {node_count}"
            if dp_size is not None:
                query += f" AND dp = {dp_size}"
            if pp_size is not None:
                query += f" AND pp = {pp_size}"
            if gpu_type is not None:
                query += f" AND gpu_type = '{gpu_type.replace("'", "''")}'"
            if max_context_length is not None:
                query += f" AND max_context_length >= {max_context_length}"
            
            # Order by updated_at to get the latest data
            query += " ORDER BY updated_at DESC LIMIT 1"
            
            result = profile_client._execute_sql_query(query)
            
            if "records" not in result or not result["records"]:
                raise ValueError(f"No prefill profile data found for model={model_name}, tp_size={tp_size}")
            
            # Extract profile_id from the result
            record = result["records"][0]
            columns = [col['name'] for col in result.get('column_metadata', [])]
            profile_id = record[columns.index('uniq_profile_id')].get('stringValue', '')
            
            # Query all data points for this profile, handling duplicates by selecting the most recent one
            data_query = f"""
            SELECT prefill_isl, prefill_ttft, prefill_throughput_per_gpu
            FROM (
                SELECT prefill_isl, prefill_ttft, prefill_throughput_per_gpu,
                       ROW_NUMBER() OVER (PARTITION BY prefill_isl ORDER BY updated_at DESC) as rn
                FROM {profile_client.table_name}
                WHERE uniq_profile_id = '{profile_id.replace("'", "''")}'
                AND prefill_isl IS NOT NULL
                AND prefill_ttft IS NOT NULL
                AND prefill_throughput_per_gpu IS NOT NULL
            ) ranked_data
            WHERE rn = 1
            ORDER BY prefill_isl
            """
            
            data_result = profile_client._execute_sql_query(data_query)
            if "records" not in data_result or not data_result["records"]:
                raise ValueError(f"No prefill data points found for profile_id={profile_id}")

            # Extract arrays
            prefill_isl_array = []
            prefill_ttft_array = []
            prefill_thpt_array = []
            
            for record in data_result["records"]:
                # Handle both stringValue and doubleValue formats
                isl_val = record[0].get('stringValue') or record[0].get('doubleValue', 0)
                ttft_val = record[1].get('stringValue') or record[1].get('doubleValue', 0)
                thpt_val = record[2].get('stringValue') or record[2].get('doubleValue', 0)
                
                prefill_isl_array.append(float(isl_val))
                prefill_ttft_array.append(float(ttft_val))
                prefill_thpt_array.append(float(thpt_val))
            
            # Additional Python-level deduplication as fallback
            unique_data = {}
            for i, isl in enumerate(prefill_isl_array):
                if isl not in unique_data:
                    unique_data[isl] = {
                        'ttft': prefill_ttft_array[i],
                        'thpt': prefill_thpt_array[i]
                    }
            
            # Rebuild arrays with unique values
            prefill_isl_array = sorted(unique_data.keys())
            prefill_ttft_array = [unique_data[isl]['ttft'] for isl in prefill_isl_array]
            prefill_thpt_array = [unique_data[isl]['thpt'] for isl in prefill_isl_array]
            
            logger.info(f"Created PrefillInterpolator with {len(prefill_isl_array)} unique data points from profile_id={profile_id}")
            
            # Check if we have enough data points for 1D interpolation
            if len(prefill_isl_array) < 2:
                raise ValueError(f"Insufficient data points for 1D interpolation. Got {len(prefill_isl_array)} points, need at least 2. "
                               f"ISL values: {prefill_isl_array}")
            
            return PrefillInterpolator(
                prefill_isl=prefill_isl_array,
                prefill_ttft=prefill_ttft_array,
                prefill_thpt_per_gpu=prefill_thpt_array
            )
            
        except Exception as e:
            logger.error(f"Failed to create PrefillInterpolator from database: {e}")
            raise
    
    else:
        raise ValueError("Either profile_results_dir or (model_name, tp_size) must be provided")


async def create_decode_interpolator(
    profile_results_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    tp_size: Optional[int] = None,
    backend: Optional[str] = None,
    gpu_count: Optional[int] = None,
    node_count: Optional[int] = None,
    dp_size: Optional[int] = None,
    pp_size: Optional[int] = None,
    gpu_type: Optional[str] = None,
    max_context_length: Optional[int] = None,
    resolution: int = 100
) -> DecodeInterpolator:
    """
    Create a DecodeInterpolator from either file-based or database-based data.
    
    Args:
        profile_results_dir: Path to profile results directory (file-based)
        model_name: Model name for database lookup
        tp_size: Tensor parallelism size
        backend: Backend name (e.g., "vllm", "trtllm", "sglang")
        gpu_count: Number of GPUs
        node_count: Number of nodes
        dp_size: Data parallelism size
        pp_size: Pipeline parallelism size
        gpu_type: GPU type (e.g., "h100", "a100")
        max_context_length: Maximum context length
        resolution: Resolution for interpolation grid
        
    Returns:
        DecodeInterpolator instance
    """
    if profile_results_dir is not None:
        # File-based initialization
        return DecodeInterpolator(profile_results_dir=profile_results_dir, resolution=resolution)
    
    elif model_name is not None and tp_size is not None:
        # Database-based initialization
        try:
            # Initialize WorkerProfileClient
            profile_client = WorkerProfileClient()
            
            # Get the latest profile data for the specified configuration
            query = f"""
            SELECT uniq_profile_id, max_kv_tokens
            FROM {profile_client.table_name}
            WHERE model_name = '{model_name.replace("'", "''")}'
            AND tp = {tp_size}
            AND worker_type = 'decode'
            """
            
            # Add optional filters
            if backend is not None:
                query += f" AND backend = '{backend.replace("'", "''")}'"
            if gpu_count is not None:
                query += f" AND gpu_count = {gpu_count}"
            if node_count is not None:
                query += f" AND node_count = {node_count}"
            if dp_size is not None:
                query += f" AND dp = {dp_size}"
            if pp_size is not None:
                query += f" AND pp = {pp_size}"
            if gpu_type is not None:
                query += f" AND gpu_type = '{gpu_type.replace("'", "''")}'"
            if max_context_length is not None:
                query += f" AND max_context_length >= {max_context_length}"
            
            # Order by updated_at to get the latest data
            query += " ORDER BY updated_at DESC LIMIT 1"
            
            result = profile_client._execute_sql_query(query)
            
            if "records" not in result or not result["records"]:
                raise ValueError(f"No decode profile data found for model={model_name}, tp_size={tp_size}")
            
            # Extract profile_id and max_kv_tokens
            record = result["records"][0]
            columns = [col['name'] for col in result.get('column_metadata', [])]
            
            profile_id = record[columns.index('uniq_profile_id')].get('stringValue', '')
            max_kv_tokens = int(record[columns.index('max_kv_tokens')].get('longValue', 0))
            
            # Get all data points for this profile_id to build the interpolator, handling duplicates
            data_query = f"""
            SELECT x_kv_usage, y_context_length, z_itl, z_thpt_per_gpu
            FROM (
                SELECT x_kv_usage, y_context_length, z_itl, z_thpt_per_gpu,
                       ROW_NUMBER() OVER (PARTITION BY x_kv_usage, y_context_length ORDER BY updated_at DESC) as rn
                FROM {profile_client.table_name}
                WHERE uniq_profile_id = '{profile_id.replace("'", "''")}'
                AND x_kv_usage IS NOT NULL
                AND y_context_length IS NOT NULL
                AND z_itl IS NOT NULL
                AND z_thpt_per_gpu IS NOT NULL
            ) ranked_data
            WHERE rn = 1
            ORDER BY x_kv_usage, y_context_length
            """
            
            data_result = profile_client._execute_sql_query(data_query)

            if "records" not in data_result or not data_result["records"]:
                raise ValueError(f"No decode data points found for profile_id={profile_id}")
            
            # Extract arrays
            x_kv_usage_array = []
            y_context_length_array = []
            z_itl_array = []
            z_thpt_array = []
            
            for record in data_result["records"]:
                # Handle both stringValue and doubleValue/longValue formats
                kv_val = record[0].get('stringValue') or record[0].get('doubleValue', 0)
                ctx_val = record[1].get('stringValue') or record[1].get('longValue', 0)
                itl_val = record[2].get('stringValue') or record[2].get('doubleValue', 0)
                thpt_val = record[3].get('stringValue') or record[3].get('doubleValue', 0)
                
                x_kv_usage_array.append(float(kv_val))
                y_context_length_array.append(int(ctx_val))
                z_itl_array.append(float(itl_val))
                z_thpt_array.append(float(thpt_val))
            
            # Additional Python-level deduplication as fallback
            unique_data = {}
            for i, kv_usage in enumerate(x_kv_usage_array):
                context_length = y_context_length_array[i]
                key = (kv_usage, context_length)
                if key not in unique_data:
                    unique_data[key] = {
                        'itl': z_itl_array[i],
                        'thpt': z_thpt_array[i]
                    }
            
            # Rebuild arrays with unique values
            x_kv_usage_array = [key[0] for key in sorted(unique_data.keys())]
            y_context_length_array = [key[1] for key in sorted(unique_data.keys())]
            z_itl_array = [unique_data[key]['itl'] for key in sorted(unique_data.keys())]
            z_thpt_array = [unique_data[key]['thpt'] for key in sorted(unique_data.keys())]
            
            logger.info(f"Created DecodeInterpolator with {len(x_kv_usage_array)} unique data points from profile_id={profile_id}")
            logger.info(f"KV usage values: {list(set(x_kv_usage_array))}")
            logger.info(f"Context length values: {list(set(y_context_length_array))}")
            
            # Check if we have enough data points for 2D interpolation
            if len(x_kv_usage_array) < 4:
                raise ValueError(f"Insufficient data points for 2D interpolation. Got {len(x_kv_usage_array)} points, need at least 4. "
                               f"KV usage values: {list(set(x_kv_usage_array))}, "
                               f"Context length values: {list(set(y_context_length_array))}")
            
            return DecodeInterpolator(
                resolution=resolution,
                x_kv_usage=x_kv_usage_array,
                y_context_length=y_context_length_array,
                z_itl=z_itl_array,
                z_thpt_per_gpu=z_thpt_array,
                max_kv_tokens=max_kv_tokens
            )
            
        except Exception as e:
            logger.error(f"Failed to create DecodeInterpolator from database: {e}")
            raise
    
    else:
        raise ValueError("Either profile_results_dir or (model_name, tp_size) must be provided")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_results_dir", type=str, required=True)
    parser.add_argument("--isl", type=int, default=3000)
    parser.add_argument("--osl", type=int, default=150)
    parser.add_argument("--ttft", type=float, default=0.1, help="in s")
    parser.add_argument("--itl", type=float, default=0.01, help="in s")
    args = parser.parse_args()

    print(f"ISL={args.isl}, OSL={args.osl}")
    print(f"TTFT={args.ttft}s, ITL={args.itl}s")
    print(f"Using profile results from {args.profile_results_dir}")
    print("")

    # first interpolate prefill
    print("Interpolating prefill performance ...")
    prefill_interpolator = PrefillInterpolator(args.profile_results_dir)

    est_ttft = prefill_interpolator.interpolate_ttft(args.isl)
    est_thpt_per_gpu = prefill_interpolator.interpolate_thpt_per_gpu(args.isl)

    if est_ttft <= args.ttft:
        print(
            f"\tEstimated TTFT={est_ttft:.3f}s <= target TTFT={args.ttft:.3f}s. Requests can queue {args.ttft - est_ttft:.3f}s maximally while meeting TTFT SLA."
        )
    else:
        print(
            f"\tEstimated TTFT={est_ttft:.3f}s > target TTFT={args.ttft:.3f}s. Cannot meet TTFT SLA."
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
            f"\tEstimated ITL={est_itl:.4f}s <= target ITL={args.itl:.4f}s at {est_kv_usage*100:.2f}% active kv usage."
        )
        print(
            f"\tEstimated throughput: {est_thpt_per_gpu:.2f} token/s/gpu. Request rate at {est_thpt_per_gpu / args.osl:.2f} requests/s will saturate one GPU."
        )
    else:
        print(
            f"\tEstimated ITL={est_itl:.4f}s > target ITL={args.itl:.4f}s. Cannot meet ITL SLA."
        )
