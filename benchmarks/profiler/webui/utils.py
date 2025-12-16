# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import queue
import threading
from enum import Enum
from pathlib import Path

import gradio as gr
import numpy as np
from aiconfigurator.webapp.components.profiling import (
    create_performance_results_section,
    create_profiling_ui_components,
    inject_profiling_assets,
    load_profiling_javascript,
)

from benchmarks.profiler.utils.defaults import DEFAULT_GPU_COST_PER_HOUR
from benchmarks.profiler.utils.pareto import compute_pareto

logger = logging.getLogger(__name__)


class PlotType(str, Enum):
    """Enum for the three plot/config types in the WebUI."""

    PREFILL = "prefill"
    DECODE = "decode"
    COST = "cost"


# Color palette for chart datasets
# TODO: handle case with more than 8 lines
CHART_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
]

# TODO: is this too long?
WEB_UI_SELECTION_TIMEOUT = 3600


def generate_config_data(
    prefill_data,
    decode_data,
    args,
    gpu_cost_per_hour: float = DEFAULT_GPU_COST_PER_HOUR,
    write_to_disk: bool = True,
):
    """
    Generate JSON data file for WebUI from profiling results.

    Args:
        prefill_data: PrefillProfileData instance
        decode_data: DecodeProfileData instance
        args: Arguments containing SLA targets (ttft, itl, isl, osl) and output_dir
        gpu_cost_per_hour: GPU cost in $/GPU/hour used for cost plot/table
        write_to_disk: Whether to write the generated JSON to args.output_dir/webui_data.json

    Returns:
        dict: Data dict for WebUI consumption.
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
    populate_cost_data(
        data, prefill_data, decode_data, args, gpu_cost_per_hour=gpu_cost_per_hour
    )

    # Save JSON file (optional)
    if write_to_disk:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Generated WebUI config data at {output_path}")

    return data


def create_gpu_cost_update_handler(
    *,
    prefill_data,
    decode_data,
    args,
    data_dict_ref,
    default_gpu_cost_per_hour: float = DEFAULT_GPU_COST_PER_HOUR,
):
    """Create a Gradio change-handler that regenerates profiling JSON when GPU cost changes."""

    def update_gpu_cost_per_hour(gpu_cost_per_hour):
        try:
            gpu_cost = float(gpu_cost_per_hour)
        except Exception:
            gpu_cost = default_gpu_cost_per_hour

        new_data = generate_config_data(
            prefill_data,
            decode_data,
            args,
            gpu_cost_per_hour=gpu_cost,
            write_to_disk=False,
        )
        data_dict_ref["data"] = new_data
        return json.dumps(new_data)

    return update_gpu_cost_per_hour


def populate_prefill_data(data, prefill_data):
    """Populate prefill chart and table data."""
    if not prefill_data.num_gpus:
        return

    # Get unique GPU counts for labels
    unique_gpus = sorted(set(prefill_data.num_gpus))
    data[PlotType.PREFILL]["chart"]["labels"] = [f"{gpu} GPUs" for gpu in unique_gpus]

    # Populate chart data points
    chart_data = []
    for i, (gpu, ttft, thpt, label) in enumerate(
        zip(
            prefill_data.num_gpus,
            prefill_data.ttft,
            prefill_data.thpt_per_gpu,
            prefill_data.parallel_mapping_labels,
        )
    ):
        chart_data.append(
            {
                "x": round(ttft, 2),
                "y": round(thpt, 2),
                "gpu": gpu,
                "tableIdx": i,
                "gpuLabel": f"{gpu} GPUs [{label}]",
            }
        )
    data[PlotType.PREFILL]["chart"]["datasets"][0]["data"] = chart_data

    # Populate table data
    table_data = []
    for i, (gpu, ttft, thpt, label) in enumerate(
        zip(
            prefill_data.num_gpus,
            prefill_data.ttft,
            prefill_data.thpt_per_gpu,
            prefill_data.parallel_mapping_labels,
        )
    ):
        # TODO: Add actual config YAML data
        config_yaml = f"prefill_config_{i}.yaml"
        table_data.append([gpu, round(ttft, 2), round(thpt, 2), config_yaml])
    data[PlotType.PREFILL]["table"]["data"] = table_data


def populate_decode_data(data, decode_data):
    """Populate decode chart and table data."""
    if not decode_data.num_gpus:
        return

    # Group by GPU count for multiple datasets
    gpu_groups: dict[int, list[dict[str, float | int]]] = {}
    for i, (gpu, itl, thpt, label) in enumerate(
        zip(
            decode_data.num_gpus,
            decode_data.itl,
            decode_data.thpt_per_gpu,
            decode_data.parallel_mapping_labels,
        )
    ):
        if gpu not in gpu_groups:
            gpu_groups[gpu] = []
        gpu_groups[gpu].append({"x": round(itl, 2), "y": round(thpt, 2), "tableIdx": i})

    # Create datasets for each GPU count with different colors
    datasets = []
    for idx, (gpu, points) in enumerate(sorted(gpu_groups.items())):
        color = CHART_COLORS[idx % len(CHART_COLORS)]
        datasets.append(
            {
                "label": f"{gpu} GPUs",
                "data": points,
                "backgroundColor": color,
                "borderColor": color,
            }
        )
    data[PlotType.DECODE]["chart"]["datasets"] = datasets

    # Populate table data
    table_data = []
    for i, (gpu, itl, thpt, label) in enumerate(
        zip(
            decode_data.num_gpus,
            decode_data.itl,
            decode_data.thpt_per_gpu,
            decode_data.parallel_mapping_labels,
        )
    ):
        config_yaml = f"decode_config_{i}.yaml"
        table_data.append([gpu, round(itl, 2), round(thpt, 2), config_yaml])
    data[PlotType.DECODE]["table"]["data"] = table_data


def populate_cost_data(
    data,
    prefill_data,
    decode_data,
    args,
    gpu_cost_per_hour: float = DEFAULT_GPU_COST_PER_HOUR,
):
    """Populate cost chart and table data with pareto-optimal configurations."""
    if not prefill_data.num_gpus or not decode_data.num_gpus:
        return

    # Compute pareto front for prefill (minimize TTFT, maximize throughput)
    p_ttft, p_thpt, prefill_pareto_indices = compute_pareto(
        prefill_data.ttft, prefill_data.thpt_per_gpu
    )

    # Compute pareto front for decode (minimize ITL, maximize throughput)
    d_itl, d_thpt, decode_pareto_indices = compute_pareto(
        decode_data.itl, decode_data.thpt_per_gpu
    )

    # Convert to numpy arrays
    p_ttft = np.array(p_ttft)
    p_thpt = np.array(p_thpt)
    d_itl = np.array(d_itl)
    d_thpt = np.array(d_thpt)

    # Generate cost datasets - one line per prefill config
    cost_datasets = []
    table_data = []
    cost_index_mapping = {}  # Map cost table row idx -> (prefill_idx, decode_idx)
    table_idx = 0

    for p_idx, (_p_ttft, _p_thpt) in enumerate(zip(p_ttft, p_thpt)):
        # Calculate prefill cost (fixed for this line)
        prefill_cost = args.isl * 1000 / _p_thpt * gpu_cost_per_hour / 3600

        # For each decode config, calculate total cost
        line_data = []
        for d_idx, (_d_itl, _d_thpt) in enumerate(zip(d_itl, d_thpt)):
            # Calculate decode cost
            decode_cost = args.osl * 1000 / _d_thpt * gpu_cost_per_hour / 3600
            total_cost = prefill_cost + decode_cost

            # X-axis: tokens per user (based on ITL)
            tokens_per_user = 1000 / _d_itl

            line_data.append(
                {
                    "x": round(tokens_per_user, 2),
                    "y": round(total_cost, 2),
                    "tableIdx": table_idx,
                }
            )

            # Store mapping from cost table row to original indices
            orig_prefill_idx = prefill_pareto_indices[p_idx]
            orig_decode_idx = decode_pareto_indices[d_idx]
            cost_index_mapping[table_idx] = (orig_prefill_idx, orig_decode_idx)

            # Add to table data
            table_data.append(
                [
                    round(_p_ttft, 2),
                    round(_p_thpt, 2),
                    round(_d_itl, 2),
                    round(_d_thpt, 2),
                    round(tokens_per_user, 2),
                    round(total_cost, 2),
                    f"cost_config_{table_idx}.yaml",  # TODO: Add actual config
                ]
            )
            table_idx += 1

        # Create dataset for this prefill config
        color = CHART_COLORS[p_idx % len(CHART_COLORS)]
        cost_datasets.append(
            {
                "label": f"TTFT: {_p_ttft:.2f}ms",
                "data": line_data,
                "backgroundColor": color,
                "borderColor": color,
            }
        )

    data[PlotType.COST]["chart"]["datasets"] = cost_datasets
    data[PlotType.COST]["table"]["data"] = table_data

    # Store the index mapping in the JSON for reference
    data[PlotType.COST]["index_mapping"] = {
        str(k): list(v) for k, v in cost_index_mapping.items()
    }


def create_selection_handler(
    data_dict_ref, selection_queue, prefill_selection, decode_selection
):
    """Create a selection handler closure for the WebUI.

    Args:
        data_dict_ref: Dict wrapper holding the latest parsed JSON data (mutated when UI inputs change)
        selection_queue: Queue to communicate selections to main thread
        prefill_selection: Dict tracking prefill selection state
        decode_selection: Dict tracking decode selection state

    Returns:
        Callable: Selection handler function for Gradio
    """

    def handle_selection(selection_json):
        """Handle datapoint selection from table."""
        if not selection_json or selection_json.strip() == "":
            return

        try:
            data_dict = data_dict_ref["data"]
            selection = json.loads(selection_json)
            plot_type = selection.get("plotType")
            row_idx = selection.get("rowIndex")

            logger.info(f"Selection received: {plot_type}, row {row_idx}")

            # Store selection for later confirmation
            if plot_type == PlotType.COST:
                # Cost selection - use index mapping to get original indices
                cost_index_mapping = data_dict[PlotType.COST].get("index_mapping", {})
                mapping_entry = cost_index_mapping.get(str(row_idx))

                if mapping_entry:
                    prefill_idx, decode_idx = mapping_entry
                    if prefill_idx is not None and decode_idx is not None:
                        logger.info(
                            f"Cost selection determines: Prefill={prefill_idx}, Decode={decode_idx}"
                        )
                        # Auto-submit for cost selection
                        selection_queue.put((prefill_idx, decode_idx))
            elif plot_type == PlotType.PREFILL:
                prefill_selection["idx"] = row_idx
                logger.info(f"Prefill selected: {row_idx}")
                # Check if we have both selections
                if decode_selection["idx"] is not None:
                    logger.info(
                        f"Both selections complete: Prefill={row_idx}, Decode={decode_selection['idx']}"
                    )
                    selection_queue.put((row_idx, decode_selection["idx"]))
                else:
                    logger.info("Waiting for decode selection...")
            elif plot_type == PlotType.DECODE:
                decode_selection["idx"] = row_idx
                logger.info(f"Decode selected: {row_idx}")
                # Check if we have both selections
                if prefill_selection["idx"] is not None:
                    logger.info(
                        f"Both selections complete: Prefill={prefill_selection['idx']}, Decode={row_idx}"
                    )
                    selection_queue.put((prefill_selection["idx"], row_idx))
                else:
                    logger.info("Waiting for prefill selection...")

        except Exception as e:
            logger.error(f"Error handling selection: {e}")

    return handle_selection


def create_gradio_interface(
    json_data_str,
    handle_selection,
    update_json_data_fn=None,
    default_gpu_cost_per_hour: float = DEFAULT_GPU_COST_PER_HOUR,
):
    """Create the Gradio interface for configuration selection.

    Args:
        json_data_str: JSON string containing profiling data
        handle_selection: Selection handler function
        update_json_data_fn: Optional function that takes (gpu_cost_per_hour) and returns updated JSON string.
        default_gpu_cost_per_hour: Default GPU cost per hour used to initialize the input box.

    Returns:
        gr.Blocks: Configured Gradio demo
    """
    with gr.Blocks(title="Configuration Selection") as demo:
        # Create hidden UI components (reused from AIC profiling module)
        ui_components = create_profiling_ui_components()
        selection_input = ui_components["selection_input"]
        selection_button = ui_components["selection_button"]
        json_data = ui_components["json_data"]

        # Inject CSS and modal (reused from AIC profiling module)
        inject_profiling_assets()

        gr.Markdown("# 📊 Profiling Results - Select Configuration")

        gr.Markdown(
            """
            **Two ways to select prefill and decode configs:**
            1. **Cost Analysis** (recommended): Click any row in the Cost Analysis table - automatically determines both prefill and decode
            2. **Individual**: Click one row in the Prefill table AND one row in the Decode table
            The selection will be processed automatically once complete.

            > 📝 **Note:** The dotted red line in the prefill and decode charts are default TTFT and ITL SLAs if not specified.

            > ⚠️ **Warning:** The TTFT values here represent the ideal case when requests arrive uniformly, minimizing queueing. Real-world TTFT may be higher than profiling results. To mitigate the issue, planner uses ][correction factors](https://github.com/ai-dynamo/dynamo/blob/main/docs/planner/sla_planner.md#2-correction-factor-calculation) to adjust dynamically at runtime.
            """
        )

        with gr.Row():
            gpu_cost_per_hour = gr.Number(
                label="GPU cost per hour ($/GPU/hour)",
                value=default_gpu_cost_per_hour,
                minimum=0,
                precision=4,
            )
        if update_json_data_fn is not None:
            gpu_cost_per_hour.change(
                fn=update_json_data_fn,
                inputs=[gpu_cost_per_hour],
                outputs=[json_data],
            )

        # Performance Results Section (reused from AIC profiling module)
        create_performance_results_section()

        # Handle selection button
        selection_button.click(
            fn=handle_selection,
            inputs=[selection_input],
            outputs=[],
        )

        # Trigger visualization when JSON data changes
        json_data.change(
            fn=None,
            inputs=[json_data],
            outputs=[],
            js=(
                "(data) => { if (data && data.trim() && window.initializeVisualizations) "
                "window.initializeVisualizations(data); }"
            ),
        )

        # Load JavaScript and data automatically on page load
        def load_data():
            """Load profiling data."""
            return json_data_str

        demo.load(
            fn=load_data, inputs=[], outputs=[json_data], js=load_profiling_javascript()
        )

    return demo


def wait_for_selection(demo, selection_queue, port):
    """Launch the demo and wait for user selection.

    Args:
        demo: Gradio demo instance
        selection_queue: Queue to receive selection from UI
        port: Port number for the WebUI

    Returns:
        tuple[int, int]: (selected_prefill_idx, selected_decode_idx)
    """

    # Launch the interface in a separate thread
    def launch_thread():
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False,
            prevent_thread_lock=True,
        )

    thread = threading.Thread(target=launch_thread, daemon=True)
    thread.start()

    logger.info(f"WebUI launched. Waiting for user selection on http://0.0.0.0:{port}")
    logger.info("Please select a row from the Cost Analysis table")

    # Block and wait for selection
    try:
        selected_prefill_idx, selected_decode_idx = selection_queue.get(
            timeout=WEB_UI_SELECTION_TIMEOUT
        )
        logger.info(
            f"User selected: Prefill={selected_prefill_idx}, Decode={selected_decode_idx}"
        )

        # Close the demo
        demo.close()

        return selected_prefill_idx, selected_decode_idx

    except queue.Empty:
        logger.error("Selection timeout - no selection made within 1 hour")
        demo.close()
        # Return default
        return 0, 0
