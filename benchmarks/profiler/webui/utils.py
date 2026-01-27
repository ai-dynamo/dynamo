# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import queue
import threading
import time
from enum import Enum
from pathlib import Path

import gradio as gr
import numpy as np
import yaml
from aiconfigurator.webapp.components.profiling import (
    create_performance_results_section,
    create_profiling_ui_components,
    inject_profiling_assets,
    load_profiling_javascript,
)

from benchmarks.profiler.utils.dgd_generation import (
    generate_decode_service_config_preview,
    generate_prefill_decode_services_config_preview,
    generate_prefill_service_config_preview,
)
from benchmarks.profiler.utils.pareto import compute_pareto

logger = logging.getLogger(__name__)


# Global variable to track selection completion for graceful shutdown
_selection_complete = threading.Event()

# Global error state for propagating profiling errors to WebUI
_profiling_errors: list[str] = []


def add_profiling_error(error_message: str) -> None:
    """Add an error message to be displayed in the WebUI.

    Args:
        error_message: The error message to display
    """
    _profiling_errors.append(error_message)
    logger.error(f"Profiling error: {error_message}")


def get_profiling_errors() -> list[str]:
    """Get all profiling errors.

    Returns:
        List of error messages
    """
    return _profiling_errors.copy()


def clear_profiling_errors() -> None:
    """Clear all profiling errors."""
    _profiling_errors.clear()


def dump_yaml_with_header(header_lines: list[str], obj: dict) -> str:
    """Dump YAML with a leading comment header (used for WebUI config previews)."""
    header = "\n".join(header_lines + ["#"])
    body = yaml.safe_dump(obj, sort_keys=False)
    return f"{header}\n{body}"


def _maybe_add_model_backend_header_lines(header_lines: list[str], args) -> None:
    model = getattr(args, "model", None)
    backend = getattr(args, "backend", None)
    if model:
        header_lines.append(f"# Model: {model}")
    if backend:
        header_lines.append(f"# Backend: {backend}")


def build_single_service_preview_header_lines(
    *,
    service_name: str,
    engine_type: str,
    mapping,
    ttft_or_itl_ms: float | None,
    thpt_per_gpu: float | None,
    args,
) -> list[str]:
    header_lines = [
        "# DynamoGraphDeployment Service Config Preview",
        f"# Service: {service_name}",
        f"# Engine: {engine_type}",
        f"# Num GPUs: {mapping.get_num_gpus()}",
        f"# Parallelization: {mapping.label()}",
    ]
    if engine_type == "prefill" and ttft_or_itl_ms is not None:
        header_lines.append(f"# Profiled TTFT: {round(ttft_or_itl_ms, 2)} ms")
    if engine_type == "decode" and ttft_or_itl_ms is not None:
        header_lines.append(f"# Profiled ITL: {round(ttft_or_itl_ms, 2)} ms")
    if thpt_per_gpu is not None:
        header_lines.append(
            f"# Profiled Throughput: {round(thpt_per_gpu, 2)} tokens/s/GPU"
        )
    _maybe_add_model_backend_header_lines(header_lines, args)
    header_lines.append(
        "# Note: This is a service-only preview. Final config includes planner."
    )
    return header_lines


def build_two_service_preview_header_lines(
    *,
    prefill_service_name: str,
    decode_service_name: str,
    prefill_mapping,
    decode_mapping,
    prefill_ttft_ms: float | None,
    prefill_thpt_per_gpu: float | None,
    decode_itl_ms: float | None,
    decode_thpt_per_gpu: float | None,
    args,
) -> list[str]:
    header_lines = [
        "# DynamoGraphDeployment Services Config Preview",
        f"# Prefill service: {prefill_service_name} ({prefill_mapping.get_num_gpus()} GPU(s), {prefill_mapping.label()})",
        f"# Decode service: {decode_service_name} ({decode_mapping.get_num_gpus()} GPU(s), {decode_mapping.label()})",
    ]
    if prefill_ttft_ms is not None:
        header_lines.append(f"# Profiled TTFT: {round(prefill_ttft_ms, 2)} ms")
    if decode_itl_ms is not None:
        header_lines.append(f"# Profiled ITL: {round(decode_itl_ms, 2)} ms")
    if prefill_thpt_per_gpu is not None:
        header_lines.append(
            f"# Profiled Prefill Throughput: {round(prefill_thpt_per_gpu, 2)} tokens/s/GPU"
        )
    if decode_thpt_per_gpu is not None:
        header_lines.append(
            f"# Profiled Decode Throughput: {round(decode_thpt_per_gpu, 2)} tokens/s/GPU"
        )
    _maybe_add_model_backend_header_lines(header_lines, args)
    header_lines.append(
        "# Note: This is a services-only preview. Final config includes planner."
    )
    return header_lines


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
    write_to_disk: bool = True,
):
    """
    Generate JSON data file for WebUI from profiling results.

    Note: This function computes GPU hours (not cost). The frontend handles
    cost calculation when the user provides a GPU cost per hour value.

    Args:
        prefill_data: PrefillProfileData instance
        decode_data: DecodeProfileData instance
        args: Arguments containing SLA targets (ttft, itl, isl, osl) and output_dir
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
    ] = f"GPU Hours Per 1000 i{args.isl}o{args.osl} requests"

    # Populate data sections
    populate_prefill_data(data, prefill_data, args)
    populate_decode_data(data, decode_data, args)
    populate_cost_data(data, prefill_data, decode_data, args)

    # Save JSON file (optional)
    if write_to_disk:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Generated WebUI config data at {output_path}")

    return data


def populate_prefill_data(data, prefill_data, args):
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
    for i, (gpu, ttft, thpt, label, mapping) in enumerate(
        zip(
            prefill_data.num_gpus,
            prefill_data.ttft,
            prefill_data.thpt_per_gpu,
            prefill_data.parallel_mapping_labels,
            prefill_data.parallel_mappings,
        )
    ):
        config_obj = generate_prefill_service_config_preview(
            config_path=args.config,
            args=args,
            best_prefill_mapping=mapping,
            num_gpus_per_node=getattr(args, "num_gpus_per_node", 8),
        )
        service_name = next(iter(config_obj.keys()))
        header_lines = build_single_service_preview_header_lines(
            service_name=service_name,
            engine_type="prefill",
            mapping=mapping,
            ttft_or_itl_ms=ttft,
            thpt_per_gpu=thpt,
            args=args,
        )
        config_yaml = dump_yaml_with_header(header_lines, config_obj)
        table_data.append([gpu, round(ttft, 2), round(thpt, 2), config_yaml])
    data[PlotType.PREFILL]["table"]["data"] = table_data


def populate_decode_data(data, decode_data, args):
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
    for i, (gpu, itl, thpt, label, mapping) in enumerate(
        zip(
            decode_data.num_gpus,
            decode_data.itl,
            decode_data.thpt_per_gpu,
            decode_data.parallel_mapping_labels,
            decode_data.parallel_mappings,
        )
    ):
        config_obj = generate_decode_service_config_preview(
            config_path=args.config,
            args=args,
            best_decode_mapping=mapping,
            num_gpus_per_node=getattr(args, "num_gpus_per_node", 8),
        )
        service_name = next(iter(config_obj.keys()))
        header_lines = build_single_service_preview_header_lines(
            service_name=service_name,
            engine_type="decode",
            mapping=mapping,
            ttft_or_itl_ms=itl,
            thpt_per_gpu=thpt,
            args=args,
        )
        config_yaml = dump_yaml_with_header(header_lines, config_obj)
        table_data.append([gpu, round(itl, 2), round(thpt, 2), config_yaml])
    data[PlotType.DECODE]["table"]["data"] = table_data


def populate_cost_data(
    data,
    prefill_data,
    decode_data,
    args,
):
    """Populate cost chart and table data with pareto-optimal configurations.

    Note: This function computes GPU hours (not cost). The frontend handles
    cost calculation when the user provides a GPU cost per hour value.
    """
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
        # Get prefill config details for this pareto point
        orig_prefill_idx = prefill_pareto_indices[p_idx]
        prefill_mapping = prefill_data.parallel_mappings[orig_prefill_idx]
        prefill_num_gpus = prefill_mapping.get_num_gpus()

        # Calculate prefill GPU hours per 1000 requests
        # GPU hours = (tokens_per_request * num_requests) / (tokens_per_second_per_gpu * 3600) * num_gpus
        prefill_gpu_hours = args.isl * 1000 / _p_thpt / 3600 * prefill_num_gpus

        # For each decode config, calculate total GPU hours
        line_data = []
        for d_idx, (_d_itl, _d_thpt) in enumerate(zip(d_itl, d_thpt)):
            # Get decode config details for this pareto point
            orig_decode_idx = decode_pareto_indices[d_idx]
            decode_mapping = decode_data.parallel_mappings[orig_decode_idx]
            decode_num_gpus = decode_mapping.get_num_gpus()

            # Calculate decode GPU hours per 1000 requests (scaled by num_gpus)
            decode_gpu_hours = args.osl * 1000 / _d_thpt / 3600 * decode_num_gpus
            total_gpu_hours = prefill_gpu_hours + decode_gpu_hours

            # X-axis: tokens per user (based on ITL)
            tokens_per_user = 1000 / _d_itl

            line_data.append(
                {
                    "x": round(tokens_per_user, 2),
                    "y": round(total_gpu_hours, 4),
                    "tableIdx": table_idx,
                }
            )

            # Store mapping from cost table row to original indices
            cost_index_mapping[table_idx] = (orig_prefill_idx, orig_decode_idx)

            services_obj = generate_prefill_decode_services_config_preview(
                config_path=args.config,
                args=args,
                best_prefill_mapping=prefill_mapping,
                best_decode_mapping=decode_mapping,
                num_gpus_per_node=getattr(args, "num_gpus_per_node", 8),
            )
            # Determine service names (backend-dependent)
            service_names = list(services_obj.keys())
            # Prefer stable names by picking based on subComponentType if present; fallback to insertion order.
            prefill_service_name = service_names[0]
            decode_service_name = (
                service_names[1] if len(service_names) > 1 else service_names[0]
            )
            header_lines = build_two_service_preview_header_lines(
                prefill_service_name=prefill_service_name,
                decode_service_name=decode_service_name,
                prefill_mapping=prefill_mapping,
                decode_mapping=decode_mapping,
                prefill_ttft_ms=float(_p_ttft),
                prefill_thpt_per_gpu=float(_p_thpt),
                decode_itl_ms=float(_d_itl),
                decode_thpt_per_gpu=float(_d_thpt),
                args=args,
            )
            config_yaml = dump_yaml_with_header(header_lines, services_obj)

            # Add to table data (GPU hours, not cost - frontend handles cost conversion)
            table_data.append(
                [
                    round(_p_ttft, 2),
                    round(_p_thpt, 2),
                    round(_d_itl, 2),
                    round(_d_thpt, 2),
                    round(tokens_per_user, 2),
                    round(total_gpu_hours, 4),
                    config_yaml,
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
        Callable: Selection handler function for Gradio that returns a status message
    """

    def handle_selection(selection_json):
        """Handle datapoint selection from table.

        Returns:
            str: Status message to display in the UI
        """
        if not selection_json or selection_json.strip() == "":
            return ""

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
                        # Signal selection complete and put in queue
                        _selection_complete.set()
                        selection_queue.put((prefill_idx, decode_idx))
                        return f"✅ Configuration selected! Prefill config #{prefill_idx}, Decode config #{decode_idx}. Processing..."
            elif plot_type == PlotType.PREFILL:
                prefill_selection["idx"] = row_idx
                logger.info(f"Prefill selected: {row_idx}")
                # Check if we have both selections
                if decode_selection["idx"] is not None:
                    logger.info(
                        f"Both selections complete: Prefill={row_idx}, Decode={decode_selection['idx']}"
                    )
                    _selection_complete.set()
                    selection_queue.put((row_idx, decode_selection["idx"]))
                    return f"✅  Configuration selected! Prefill config #{row_idx}, Decode config #{decode_selection['idx']}. Processing..."
                else:
                    return f"ℹ️  Prefill config #{row_idx} selected. Please select a Decode configuration."
            elif plot_type == PlotType.DECODE:
                decode_selection["idx"] = row_idx
                logger.info(f"Decode selected: {row_idx}")
                # Check if we have both selections
                if prefill_selection["idx"] is not None:
                    logger.info(
                        f"Both selections complete: Prefill={prefill_selection['idx']}, Decode={row_idx}"
                    )
                    _selection_complete.set()
                    selection_queue.put((prefill_selection["idx"], row_idx))
                    return f"✅  Configuration selected! Prefill config #{prefill_selection['idx']}, Decode config #{row_idx}. Processing..."
                else:
                    return f"ℹ️  Decode config #{row_idx} selected. Please select a Prefill configuration."

            return ""

        except Exception as e:
            logger.error(f"Error handling selection: {e}")
            return f"❌  Error: {str(e)}"

    return handle_selection


def create_gradio_interface(
    json_data_str,
    handle_selection,
):
    """Create the Gradio interface for configuration selection.

    Args:
        json_data_str: JSON string containing profiling data
        handle_selection: Selection handler function

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

        # Display any profiling errors/warnings at the top
        profiling_errors = get_profiling_errors()
        if profiling_errors:
            error_text = "\n".join(f"- {err}" for err in profiling_errors)
            gr.Markdown(
                f"""
                <div style="background-color: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; padding: 10px; margin-bottom: 10px;">
                <strong>⚠️ Profiling Warnings/Errors:</strong>

{error_text}
                </div>
                """
            )

        gr.Markdown(
            """
            **Two ways to select prefill and decode configs:**
            1. **GPU Hours Analysis** (recommended): Select any row in the GPU Hours table - automatically determines both prefill and decode
            2. **Individual**: Select one row in the Prefill table AND one row in the Decode table
            The selection will be processed automatically once complete.

            **Chart Reference Points:** 🔴 Max Throughput Under SLA · 🟡 Max Throughput Overall · 🟢 Latency-Optimized (lowest latency under SLA)

            > 📝 **Note:** The dotted red line in the prefill and decode charts are default TTFT and ITL SLAs if not specified.

            > ⚠️ **Warning:** The TTFT values here represent the ideal case when requests arrive uniformly, minimizing queueing. Real-world TTFT may be higher than profiling results. To mitigate the issue, planner uses [correction factors](https://github.com/ai-dynamo/dynamo/blob/main/docs/planner/sla_planner.md#2-correction-factor-calculation) to adjust dynamically at runtime.

            > 💡 **Tip:** Use the GPU cost checkbox and input in the charts section to convert GPU hours to cost.
            """
        )

        # Status message display for selection feedback
        selection_status = gr.Markdown(
            value="",
            elem_id="selection_status",
        )

        # Performance Results Section (reused from AIC profiling module)
        create_performance_results_section()

        # Handle selection button - now returns status message
        selection_button.click(
            fn=handle_selection,
            inputs=[selection_input],
            outputs=[selection_status],
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

    # Reset the selection complete event
    _selection_complete.clear()

    # Block and wait for selection
    try:
        selected_prefill_idx, selected_decode_idx = selection_queue.get(
            timeout=WEB_UI_SELECTION_TIMEOUT
        )
        logger.info(
            f"User selected: Prefill={selected_prefill_idx}, Decode={selected_decode_idx}"
        )

        # Wait for the selection handler to complete and give UI time to show success message
        if _selection_complete.wait(timeout=2.0):
            # Give extra time for the UI to display the success message
            time.sleep(1.0)

        # Close the demo gracefully
        demo.close()

        return selected_prefill_idx, selected_decode_idx

    except queue.Empty:
        logger.error("Selection timeout - no selection made within 1 hour")
        demo.close()
        # Return default
        return 0, 0
