# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import queue
import threading
from pathlib import Path

import gradio as gr
import numpy as np
from aiconfigurator.webapp.components.profiling import (
    create_performance_results_section,
    create_profiling_ui_components,
    inject_profiling_assets,
    load_profiling_javascript,
)

from benchmarks.profiler.utils.defaults import GPU_COST_PER_HOUR
from benchmarks.profiler.utils.pareto import compute_pareto

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

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


def generate_config_data(prefill_data, decode_data, args):
    """
    Generate JSON data file for WebUI from profiling results.

    Args:
        prefill_data: PrefillProfileData instance
        decode_data: DecodeProfileData instance
        args: Arguments containing SLA targets (ttft, itl, isl, osl) and output_dir
    """
    # Load template
    template_path = Path(__file__).parent / "data_template.json"
    with open(template_path, "r") as f:
        data = json.load(f)

    # Construct output path
    output_path = os.path.join(args.output_dir, "webui_data.json")

    # Set SLA targets
    data["prefill"]["chart"]["target_line"]["value"] = args.ttft
    data["prefill"]["chart"]["target_line"]["label"] = f"Target TTFT: {args.ttft} ms"

    data["decode"]["chart"]["target_line"]["value"] = args.itl
    data["decode"]["chart"]["target_line"]["label"] = f"Target ITL: {args.itl} ms"

    data["cost"]["chart"]["title"] = f"Cost Per 1000 i{args.isl}o{args.osl} requests"

    # Populate prefill data
    if prefill_data.num_gpus:
        # Get unique GPU counts for labels
        unique_gpus = sorted(set(prefill_data.num_gpus))
        data["prefill"]["chart"]["labels"] = [f"{gpu} GPUs" for gpu in unique_gpus]

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
        data["prefill"]["chart"]["datasets"][0]["data"] = chart_data

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
        data["prefill"]["table"]["data"] = table_data

    # Populate decode data
    if decode_data.num_gpus:
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
            gpu_groups[gpu].append(
                {"x": round(itl, 2), "y": round(thpt, 2), "tableIdx": i}
            )

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
        data["decode"]["chart"]["datasets"] = datasets

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
            # TODO: Add actual config YAML data
            config_yaml = f"decode_config_{i}.yaml"
            table_data.append([gpu, round(itl, 2), round(thpt, 2), config_yaml])
        data["decode"]["table"]["data"] = table_data

    # Populate cost data
    cost_index_mapping = {}  # Map cost table row idx -> (prefill_idx, decode_idx)

    if prefill_data.num_gpus and decode_data.num_gpus:
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
        table_idx = 0

        for p_idx, (_p_ttft, _p_thpt) in enumerate(zip(p_ttft, p_thpt)):
            # Calculate prefill cost (fixed for this line)
            prefill_cost = args.isl * 1000 / _p_thpt * GPU_COST_PER_HOUR / 3600

            # For each decode config, calculate total cost
            line_data = []
            for d_idx, (_d_itl, _d_thpt) in enumerate(zip(d_itl, d_thpt)):
                # Calculate decode cost
                decode_cost = args.osl * 1000 / _d_thpt * GPU_COST_PER_HOUR / 3600
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

        data["cost"]["chart"]["datasets"] = cost_datasets
        data["cost"]["table"]["data"] = table_data

        # Store the index mapping in the JSON for reference
        data["cost"]["index_mapping"] = {
            str(k): list(v) for k, v in cost_index_mapping.items()
        }

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

    def handle_selection(selection_json):
        """Handle datapoint selection from table."""
        if not selection_json or selection_json.strip() == "":
            return

        try:
            selection = json.loads(selection_json)
            plot_type = selection.get("plotType")
            row_idx = selection.get("rowIndex")

            logger.info(f"Selection received: {plot_type}, row {row_idx}")

            # Store selection for later confirmation
            if plot_type == "cost":
                # Cost selection - use index mapping to get original indices
                cost_index_mapping = data_dict["cost"].get("index_mapping", {})
                mapping_entry = cost_index_mapping.get(str(row_idx))

                if mapping_entry:
                    prefill_idx, decode_idx = mapping_entry
                    if prefill_idx is not None and decode_idx is not None:
                        logger.info(
                            f"Cost selection determines: Prefill={prefill_idx}, Decode={decode_idx}"
                        )
                        # Auto-submit for cost selection
                        selection_queue.put((prefill_idx, decode_idx))
            elif plot_type == "prefill":
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
            elif plot_type == "decode":
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

    # Create Gradio interface
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
            2. **Individual**: Click one row in Prefill table AND one row in Decode table
            The selection will be processed automatically once complete.

            > 📝 **Note:** The dotted red line in the prefill and decode charts are default TTFT and ITL SLAs if not specified.

            > ⚠️ **Warning:** The TTFT values here represent the ideal case when requests arrive uniformly, minimizing queueing. Real-world TTFT may be higher than profiling results. To mitigate the issue, turn on correction factor in planner.
            """
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

    # Launch the interface in a separate thread
    def launch_thread():
        demo.launch(
            server_name="0.0.0.0",
            server_port=args.webui_port,
            share=False,
            prevent_thread_lock=True,
        )

    thread = threading.Thread(target=launch_thread, daemon=True)
    thread.start()

    logger.info(
        f"WebUI launched. Waiting for user selection on http://0.0.0.0:{args.webui_port}"
    )
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
