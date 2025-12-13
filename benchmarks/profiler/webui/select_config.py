# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import queue

from benchmarks.profiler.utils.defaults import DEFAULT_GPU_COST_PER_HOUR
from benchmarks.profiler.webui.utils import (
    create_gpu_cost_update_handler,
    create_gradio_interface,
    create_selection_handler,
    generate_config_data,
    wait_for_selection,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def pick_config_with_webui(prefill_data, decode_data, args):
    """
    Launch WebUI for user to pick configurations.

    Args:
        prefill_data: PrefillProfileData instance
        decode_data: DecodeProfileData instance
        args: Arguments containing SLA targets and output_dir

    Returns:
        tuple[int, int]: (selected_prefill_idx, selected_decode_idx)
    """
    # Generate JSON data (also writes default JSON file for convenience)
    data_dict = generate_config_data(
        prefill_data,
        decode_data,
        args,
        gpu_cost_per_hour=DEFAULT_GPU_COST_PER_HOUR,
        write_to_disk=True,
    )
    json_data_str = json.dumps(data_dict)

    logger.info(f"Launching WebUI on port {args.webui_port}...")

    # Queue to communicate selection from UI to main thread
    selection_queue: queue.Queue[tuple[int | None, int | None]] = queue.Queue()

    # Track individual selections
    prefill_selection = {"idx": None}
    decode_selection = {"idx": None}

    # Create selection handler and Gradio interface
    data_dict_ref = {"data": data_dict}
    handle_selection = create_selection_handler(
        data_dict_ref, selection_queue, prefill_selection, decode_selection
    )
    update_gpu_cost_per_hour = create_gpu_cost_update_handler(
        prefill_data=prefill_data,
        decode_data=decode_data,
        args=args,
        data_dict_ref=data_dict_ref,
        default_gpu_cost_per_hour=DEFAULT_GPU_COST_PER_HOUR,
    )

    demo = create_gradio_interface(
        json_data_str,
        handle_selection,
        update_json_data_fn=update_gpu_cost_per_hour,
        default_gpu_cost_per_hour=DEFAULT_GPU_COST_PER_HOUR,
    )

    return wait_for_selection(demo, selection_queue, args.webui_port)
