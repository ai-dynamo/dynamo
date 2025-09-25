# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class NpzFile:
    """Handler for NumPy compressed archive (.npz) files containing performance data.

    Provides utilities to load, manipulate, and save benchmark results
    stored in NumPy's compressed archive format.

    Attributes:
        data: Dictionary containing loaded arrays from the npz file
        npz_file_path: Path to the source npz file
    """

    def __init__(self, npz_file_path):
        """Load data from an npz file into memory.

        Args:
            npz_file_path: Path to the .npz file to load
        """
        with np.load(npz_file_path) as loaded_data:
            self.data = {key: loaded_data[key] for key in loaded_data.keys()}
        self.npz_file_path = npz_file_path

    def show_npz_data(self):
        """Display detailed information about the contents of the npz file.

        Prints keys, shapes, data types, sizes, and sample values for each array.
        """
        print("NPZ file contents:")
        print(f"Keys in the file: {list(self.data.keys())}")
        print()
        # Display information about each array in the file
        for key in self.data.keys():
            array = self.data[key]
            print(f"Key: {key}")
            print(f"  Shape: {array.shape}")
            print(f"  Data type: {array.dtype}")
            print(f"  Size: {array.size}")
            if array.size <= 20:  # Only print values for small arrays
                print(f"  Values: {array}")
            else:
                print(
                    f"  Min: {np.min(array)}, Max: {np.max(array)}, Mean: {np.mean(array):.4f}",
                )
            print()

    def save_npz_file(
        self,
        data: Optional[dict] = None,
        output_path: Optional[Path] = None,
        compressed=True,
    ):
        """Save data to an npz file.

        Args:
            data: Dictionary of arrays to save. Uses self.data if None
            output_path: Output file path. Uses self.npz_file_path if None
            compressed: Whether to use compressed format (npz vs npz_compressed)
        """
        if data is None:
            data = self.data
        if output_path is None:
            output_path = self.npz_file_path
        if compressed:
            np.savez_compressed(output_path, **data)
        else:
            np.savez(output_path, **data)
        logger.info(f"Data saved to {output_path}")


class MergedNpz(NpzFile):
    """Extended NpzFile for merging multiple performance benchmark files.

    Handles combining multiple npz files with potentially different schemas
    into a unified format suitable for performance interpolation.

    Attributes:
        row_count: Number of data rows in the merged file
        column_count: Number of columns/keys in the data
    """

    def __init__(self, npz_file_path):
        """Initialize merged npz file, converting 1D arrays to 2D format.

        Args:
            npz_file_path: Path to the npz file to load and prepare for merging
        """
        super().__init__(npz_file_path)
        updated = False
        for key in self.data.keys():
            array = self.data[key]
            if array.ndim == 1:
                self.data[key] = np.array([array])
                updated = True
        if updated:
            self.save_npz_file()
        self.row_count = self.data[list(self.data.keys())[0]].shape[0]
        self.column_count = len(self.data.keys())

    def merge(self, other_npz_file_path):
        """Merge another npz file into this one.

        Args:
            other_npz_file_path: Path to npz file to merge in
        """
        other_npz = NpzFile(other_npz_file_path)
        for key in other_npz.data.keys():
            # shape (row_count, x) becomes (row_count + 1, x)
            if key not in self.data.keys():
                self.data[key] = np.full((self.row_count,), None, dtype=object)
            self.data[key] = np.vstack(
                (self.data[key], np.array([other_npz.data[key]]))
            )
        for key in self.data.keys():
            if key not in other_npz.data.keys():
                self.data[key] = np.full((self.row_count,), None, dtype=object)
        self.save_npz_file()


PRE_SWEPT_CONFIG_KEYS = [
    "gpu_type",
    "model",
    "framework",
    "framework_version",
    "tp",
    "dp",
    "pp",
    "block_size",
    "max_batch_size",
    "gpu_count",
]


class PrefillNpz(NpzFile):
    """Specialized NpzFile for prefill performance benchmark data.

    Handles prefill-specific performance metrics including throughput per GPU,
    time to first token (TTFT), and input sequence length relationships.
    """

    COLUMNS = PRE_SWEPT_CONFIG_KEYS + [
        "prefill_isl",
        "prefill_ttft",
        "prefill_thpt_per_gpu",
    ]

    def __init__(self, npz_file_path, configs: dict):
        """Initialize prefill npz file with configuration validation.

        Args:
            npz_file_path: Path to the prefill performance data file
            configs: Configuration dictionary containing system parameters
        """
        super().__init__(npz_file_path)
        updated = False
        for col in self.COLUMNS:
            # each row should include all the pre_swept_configs
            if col not in self.data.keys():
                assert (
                    configs is not None and col in configs
                ), f"Column {col} not found in pre_swept_configs: {configs}"
                self.data[col] = np.array([configs[col]])
                updated = True
        if updated:
            self.save_npz_file()


class DecodeNpz(NpzFile):
    """Specialized NpzFile for decode performance benchmark data.

    Handles decode-specific performance metrics including inter-token latency (ITL),
    context length effects, and KV cache usage patterns.
    """

    COLUMNS = PRE_SWEPT_CONFIG_KEYS + [
        "x_kv_usage",
        "y_context_length",
        "z_itl",
        "z_thpt_per_gpu",
        "max_kv_tokens",
    ]

    def __init__(self, npz_file_path, configs: dict):
        """Initialize decode npz file with configuration validation.

        Args:
            npz_file_path: Path to the decode performance data file
            configs: Configuration dictionary containing system parameters
        """
        super().__init__(npz_file_path)
        updated = False
        for col in self.COLUMNS:
            if col not in self.data.keys():
                assert (
                    configs is not None and col in configs
                ), f"Column {col} not found in pre_swept_configs: {configs}"
                self.data[col] = np.array([configs[col]])
                updated = True
        if updated:
            self.save_npz_file()


def merge_raw_data(raw_data_npz_path, configs, mode):
    """Merge raw performance data into the consolidated pre-swept results folder.

    Args:
        raw_data_npz_path: Path to raw benchmark data file
        configs: System configuration parameters
        mode: Either "prefill" or "decode" to specify data type

    Raises:
        ValueError: If mode is not "prefill" or "decode"
    """
    # check if mode is valid
    # convert raw_data_file to prefill or decode npz file
    if mode == "prefill":
        PrefillNpz(raw_data_npz_path, configs)
    elif mode == "decode":
        DecodeNpz(raw_data_npz_path, configs)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # merge the npz file
    merged_npz_path = f'pre_swept_results/{configs["gpu_type"]}/{configs["framework"]}/{configs["model"]}/{mode}.npz'
    os.makedirs(os.path.dirname(merged_npz_path), exist_ok=True)
    if not os.path.exists(merged_npz_path):
        # copy the raw data npz file to the merged npz file and add one dimension.
        shutil.copy(raw_data_npz_path, merged_npz_path)
        MergedNpz(merged_npz_path)
        return
    merged_npz = MergedNpz(merged_npz_path)
    merged_npz.merge(raw_data_npz_path)


class PreSweptResultsHelper:
    """Helper class for retrieving pre-swept performance data.

    Provides interface to access pre-computed performance benchmarks
    for specific hardware and model configurations without running
    new profiling experiments.

    Attributes:
        gpu_type: GPU hardware type (e.g., "h200_sxm")
        framework: Inference framework (e.g., "vllm")
        model_name: Model identifier (e.g., "nvidia/Llama-3.1-8B-Instruct-FP8")
    """

    def __init__(self, gpu_type, framework, model_name):
        """Initialize helper for specific hardware and model configuration.

        Args:
            gpu_type: Type of GPU hardware
            framework: Inference framework name
            model_name: Model identifier string
        """
        self.gpu_type = gpu_type
        self.framework = framework
        self.model_name = model_name

    def get_data(self, mode):
        """Retrieve all performance data for the specified mode.

        Args:
            mode: Either "prefill" or "decode"

        Returns:
            dict: Dictionary containing all performance arrays for the mode
        """
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        npz_file_path = f"{current_file_dir}/pre_swept_results/{self.gpu_type}/{self.framework}/{self.model_name}/{mode}.npz"
        npz_file = NpzFile(npz_file_path)
        return npz_file.data

    def select_data(self, mode, configs):
        """Select specific performance data matching the given configuration.

        Args:
            mode: Either "prefill" or "decode"
            configs: Configuration parameters to match

        Returns:
            dict: Performance data dictionary for the matching configuration

        Raises:
            ValueError: If no matching data found or invalid config keys
        """
        all_data = self.get_data(mode)
        row_count = all_data[list(all_data.keys())[0]].shape[0]
        for row_idx in range(row_count):
            valid = True
            for config_name, config_value in configs.items():
                if config_name not in all_data.keys():
                    raise ValueError(
                        f"Column {config_name} not found in all_data: {all_data.keys()}"
                    )
                row_value = all_data[config_name][row_idx]
                # Handle both scalar and array values
                if np.isscalar(row_value):
                    if config_value != row_value:
                        valid = False
                        break
                else:
                    if config_value not in row_value:
                        valid = False
                        break
            if valid:
                return {key: all_data[key][row_idx] for key in all_data.keys()}
        raise ValueError(f"No data found for mode: {mode} and configs: {configs}")


if __name__ == "__main__":
    # demo of how to use merge_raw_data
    merge_raw_data(
        "/home/jasonzho/repo/dynamo/tests/planner/profiling_results/H200_TP1P_TP1D/selected_prefill_interpolation/raw_data.npz",
        configs={
            "gpu_type": "h200_sxm",
            "model": "nvidia/Llama-3.1-8B-Instruct-FP8",
            "framework": "vllm",
            "framework_version": "0.10.1.1",
            "tp": 1,
            "dp": 1,
            "pp": 1,
            "block_size": 128,
            "max_batch_size": 128,
            "gpu_count": 8,
        },
        mode="prefill",
    )
    merge_raw_data(
        "/home/jasonzho/repo/dynamo/tests/planner/profiling_results/H200_TP1P_TP1D/selected_decode_interpolation/raw_data.npz",
        configs={
            "gpu_type": "h200_sxm",
            "model": "nvidia/Llama-3.1-8B-Instruct-FP8",
            "framework": "vllm",
            "framework_version": "0.10.1.1",
            "tp": 1,
            "dp": 1,
            "pp": 1,
            "block_size": 128,
            "max_batch_size": 128,
            "gpu_count": 8,
        },
        mode="decode",
    )
