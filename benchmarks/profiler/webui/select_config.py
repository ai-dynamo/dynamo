# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import queue
from pathlib import Path

from benchmarks.profiler.webui.utils import (
    PlotType,
    create_gradio_interface,
    create_selection_handler,
    populate_cost_data,
    populate_decode_data,
    populate_prefill_data,
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


def generate_config_data(prefill_data, decode_data, args):
    """
    Generate JSON data file for WebUI from profiling results.

    Args:
        prefill_data: PrefillProfileData instance
        decode_data: DecodeProfileData instance
        args: Arguments containing SLA targets (ttft, itl, isl, osl) and output_dir

    Returns a JSON data file for WebUI consumption,
        see https://github.com/ai-dynamo/aiconfigurator/blob/main/src/aiconfigurator/webapp/components/profiling/standalone/sample_profiling_data.json for more details
    """
    # Load template
    template_path = Path(__file__).parent / "data_template.json"
    with open(template_path, "r") as f:
        data = json.load(f)

    # Construct output path
    output_path = os.path.join(args.output_dir, "webui_data.json")

    # Set SLA targets
    data[PlotType.PREFILL]["chart"]["target_line"]["value"] = args.ttft
    data[PlotType.PREFILL]["chart"]["target_line"][
        "label"
    ] = f"Target TTFT: {args.ttft} ms"

    data[PlotType.DECODE]["chart"]["target_line"]["value"] = args.itl
    data[PlotType.DECODE]["chart"]["target_line"][
        "label"
    ] = f"Target ITL: {args.itl} ms"

    data[PlotType.COST]["chart"][
        "title"
    ] = f"Cost Per 1000 i{args.isl}o{args.osl} requests"

    # Populate data sections
    populate_prefill_data(data, prefill_data)
    populate_decode_data(data, decode_data)
    populate_cost_data(data, prefill_data, decode_data, args)

    # Save JSON file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"Generated WebUI config data at {output_path}")
    return data


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
    # Generate JSON data file and load it
    generate_config_data(prefill_data, decode_data, args)

    output_path = os.path.join(args.output_dir, "webui_data.json")
    with open(output_path, "r") as f:
        json_data_str = f.read()
        data_dict = json.loads(json_data_str)

    logger.info(f"Launching WebUI on port {args.webui_port}...")

    # Queue to communicate selection from UI to main thread
    selection_queue: queue.Queue[tuple[int | None, int | None]] = queue.Queue()

    # Track individual selections
    prefill_selection = {"idx": None}
    decode_selection = {"idx": None}

    # Create selection handler and Gradio interface
    handle_selection = create_selection_handler(
        data_dict, selection_queue, prefill_selection, decode_selection
    )
    demo = create_gradio_interface(json_data_str, handle_selection)

    return wait_for_selection(demo, selection_queue, args.webui_port)
