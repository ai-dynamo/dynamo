# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Recording scheduler that captures vLLM scheduler behavior for Rust implementation.

This scheduler wraps the DynamoScheduler and records all inputs/outputs
in a format suitable for replay by a Rust scheduler implementation.
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from vllm.v1.core.sched.interface import SchedulerInterface


def _to_serializable(obj: Any) -> Any:
    """Convert numpy arrays and other non-serializable types to JSON-serializable forms."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


@dataclass
class RecordedIteration:
    """A single recorded iteration of the scheduler"""

    iteration: int
    schedule_output: Dict[str, Any]
    model_runner_output: Dict[str, Any]
    engine_core_outputs: Dict[str, Any]
    timestamp: float


class RecordingScheduler(SchedulerInterface):
    """
    Scheduler that records all operations for later replay.

    This scheduler forwards all operations to the underlying vLLM scheduler
    while recording the inputs and outputs for analysis and replay.
    """

    def __init__(
        self,
        *args,
        enable_recording: bool = True,
        recording_path: Optional[Path] = None,
        **kwargs,
    ):
        """
        Initialize the recording scheduler.

        Args:
            enable_recording: Whether to enable recording
            recording_path: Path to save the recording (defaults to .sandbox/recordings/)
            *args, **kwargs: Passed to the underlying scheduler
        """
        # Determine which scheduler to use based on environment variable
        scheduler_type = os.getenv("DYN_VLLM_RECORD_SCHEDULER_CLS", "dynamo").lower()

        if scheduler_type == "vllm":
            # Use vLLM's default scheduler directly
            from vllm.v1.core.sched.scheduler import Scheduler

            self._wrapped_scheduler = Scheduler(*args, **kwargs)
            print("RecordingScheduler: Using vLLM Scheduler")
        elif scheduler_type == "dynamo":
            # Use DynamoScheduler (which wraps vLLM scheduler)
            from .dynamo import DynamoScheduler

            self._wrapped_scheduler = DynamoScheduler(*args, **kwargs)
            print("RecordingScheduler: Using DynamoScheduler")
        else:
            raise ValueError(
                f"Invalid scheduler type: {scheduler_type}. "
                f"DYN_VLLM_RECORD_SCHEDULER_CLS must be 'dynamo' or 'vllm'"
            )

        self.enable_recording = enable_recording
        self.iteration = 0
        self.recordings: List[RecordedIteration] = []
        self.current_schedule_output: Optional[Dict[str, Any]] = None

        if recording_path:
            self.recording_path = Path(recording_path)
        else:
            # Default to .sandbox/recordings/ in current working directory
            self.recording_path = Path.cwd() / ".sandbox" / "recordings"

        # Create recordings directory if it doesn't exist
        if self.enable_recording:
            self.recording_path.mkdir(parents=True, exist_ok=True)
            print(f"Recording enabled. Will save to: {self.recording_path}")

    # Forward attributes that vLLM's engine core accesses directly
    @property
    def connector(self):
        """Forward connector attribute from wrapped scheduler."""
        return getattr(self._wrapped_scheduler, "connector", None)

    @property
    def ec_connector(self):
        """Forward ec_connector attribute from wrapped scheduler."""
        return getattr(self._wrapped_scheduler, "ec_connector", None)

    def schedule(self):
        """Schedule requests and record the output."""
        output = self._wrapped_scheduler.schedule()

        if self.enable_recording:
            # Convert SchedulerOutput to dict
            self.current_schedule_output = self._scheduler_output_to_dict(output)

        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        """Update from model output and record."""
        result = self._wrapped_scheduler.update_from_output(
            scheduler_output, model_runner_output
        )

        if self.enable_recording and self.current_schedule_output:
            # Record the complete iteration
            iteration = RecordedIteration(
                iteration=self.iteration,
                schedule_output=self.current_schedule_output,
                model_runner_output=self._model_runner_output_to_dict(
                    model_runner_output
                ),
                engine_core_outputs=self._engine_core_outputs_to_dict(result),
                timestamp=time.time(),
            )
            self.recordings.append(iteration)

            # Increment iteration counter
            self.iteration += 1
            self.current_schedule_output = None

            # Optionally save incrementally (every 10 iterations)
            if self.iteration % 10 == 0:
                self.save_recording(incremental=True)

        return result

    def _scheduler_output_to_dict(self, output) -> Dict[str, Any]:
        """Convert SchedulerOutput to a dictionary."""
        try:
            return _to_serializable(
                {
                    "scheduled_new_reqs": [
                        {
                            "req_id": req.req_id,
                            "prompt_token_ids": req.prompt_token_ids,
                            "block_ids": [list(blocks) for blocks in req.block_ids]
                            if req.block_ids
                            else [],
                            "num_computed_tokens": req.num_computed_tokens,
                            "mm_hashes": req.mm_hashes
                            if hasattr(req, "mm_hashes")
                            else [],
                        }
                        for req in output.scheduled_new_reqs
                    ],
                    "scheduled_cached_reqs": {
                        "req_ids": output.scheduled_cached_reqs.req_ids,
                        "resumed_from_preemption": output.scheduled_cached_reqs.resumed_from_preemption,
                        "new_token_ids": output.scheduled_cached_reqs.new_token_ids,
                        "new_block_ids": [
                            [list(blocks) for blocks in block_ids]
                            if block_ids
                            else None
                            for block_ids in output.scheduled_cached_reqs.new_block_ids
                        ],
                        "num_computed_tokens": output.scheduled_cached_reqs.num_computed_tokens,
                    },
                    "num_scheduled_tokens": dict(output.num_scheduled_tokens),
                    "total_num_scheduled_tokens": output.total_num_scheduled_tokens,
                    "scheduled_spec_decode_tokens": dict(
                        output.scheduled_spec_decode_tokens
                    ),
                    "scheduled_encoder_inputs": dict(output.scheduled_encoder_inputs),
                    "num_common_prefix_blocks": list(output.num_common_prefix_blocks),
                    "finished_req_ids": list(output.finished_req_ids),
                    "free_encoder_mm_hashes": list(output.free_encoder_mm_hashes),
                }
            )
        except Exception as e:
            print(f"Error converting SchedulerOutput: {e}")
            return {}

    def _model_runner_output_to_dict(self, output) -> Dict[str, Any]:
        """Convert ModelRunnerOutput to a dictionary."""
        try:
            result = {
                "req_ids": _to_serializable(output.req_ids),
                "req_id_to_index": _to_serializable(dict(output.req_id_to_index)),
                "sampled_token_ids": _to_serializable(output.sampled_token_ids),
            }

            if output.logprobs:
                result["logprobs"] = {
                    "logprob_token_ids": _to_serializable(
                        output.logprobs.logprob_token_ids
                    ),
                    "logprobs": _to_serializable(output.logprobs.logprobs),
                    "sampled_token_ranks": _to_serializable(
                        output.logprobs.sampled_token_ranks
                    ),
                }

            if hasattr(output, "num_nans_in_logits") and output.num_nans_in_logits:
                result["num_nans_in_logits"] = _to_serializable(
                    dict(output.num_nans_in_logits)
                )

            return result
        except Exception as e:
            print(f"Error converting ModelRunnerOutput: {e}")
            return {}

    def _engine_core_outputs_to_dict(self, outputs) -> Dict[str, Any]:
        """Convert dict[int, EngineCoreOutputs] to a serializable format."""
        try:
            # outputs is a dict mapping engine_idx -> EngineCoreOutputs
            result = {}
            for engine_idx, engine_outputs in outputs.items():
                # Each engine_outputs is an EngineCoreOutputs struct
                engine_result = {
                    "engine_index": engine_outputs.engine_index,
                    "outputs": [
                        {
                            "request_id": output.request_id,
                            "new_token_ids": _to_serializable(output.new_token_ids),
                            "finish_reason": output.finish_reason.value
                            if output.finish_reason
                            else None,
                            "num_cached_tokens": output.num_cached_tokens,
                            "stop_reason": output.stop_reason
                            if hasattr(output, "stop_reason")
                            else None,
                        }
                        for output in engine_outputs.outputs
                    ],
                    "timestamp": engine_outputs.timestamp,
                }

                # Add optional fields if present
                if engine_outputs.scheduler_stats:
                    engine_result["scheduler_stats"] = {
                        "num_running_reqs": engine_outputs.scheduler_stats.num_running_reqs,
                        "num_waiting_reqs": engine_outputs.scheduler_stats.num_waiting_reqs,
                        "num_preempted_reqs": getattr(
                            engine_outputs.scheduler_stats, "num_preempted_reqs", 0
                        ),
                    }

                if engine_outputs.finished_requests:
                    engine_result["finished_requests"] = list(
                        engine_outputs.finished_requests
                    )

                result[str(engine_idx)] = engine_result

            return _to_serializable(result)
        except Exception as e:
            print(f"Error converting EngineCoreOutputs: {e}")
            import traceback

            traceback.print_exc()
            return {}

    def save_recording(self, incremental: bool = False):
        """Save the recording to a JSON file."""
        if not self.recordings:
            print("No recordings to save")
            return

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "_incremental" if incremental else ""
        filename = f"scheduler_trace_{timestamp}{suffix}.json"
        filepath = self.recording_path / filename

        # Create the trace structure
        trace = {
            "metadata": {
                "vllm_version": "0.10.2",  # Could get this dynamically
                "model": "gpt2",  # Should be passed in
                "timestamp": datetime.now().isoformat(),
                "total_iterations": len(self.recordings),
            },
            "iterations": [asdict(rec) for rec in self.recordings],
        }

        # Save to file
        with open(filepath, "w") as f:
            json.dump(trace, f, indent=2)

        print(f"Saved {len(self.recordings)} iterations to {filepath}")

    def shutdown(self):
        """Save recording and shutdown."""
        if self.enable_recording and self.recordings:
            self.save_recording()
        self._wrapped_scheduler.shutdown()

    def add_request(self, request) -> None:
        """Add a new request to the scheduler."""
        self._wrapped_scheduler.add_request(request)

    def finish_requests(self, request_ids, finished_status) -> None:
        """Mark requests as finished."""
        self._wrapped_scheduler.finish_requests(request_ids, finished_status)

    def get_num_unfinished_requests(self) -> int:
        """Get the number of unfinished requests."""
        return self._wrapped_scheduler.get_num_unfinished_requests()

    def has_finished_requests(self) -> bool:
        """Check if there are any finished requests."""
        return self._wrapped_scheduler.has_finished_requests()

    def reset_prefix_cache(self) -> bool:
        """Reset the prefix cache."""
        return self._wrapped_scheduler.reset_prefix_cache()

    def get_request_counts(self):
        """Get counts of requests in different states."""
        return self._wrapped_scheduler.get_request_counts()

    def make_stats(self):
        """Generate statistics about the scheduler's current state."""
        return self._wrapped_scheduler.make_stats()

    def update_draft_token_ids(self, draft_token_ids) -> None:
        """Update draft token IDs for scheduled requests."""
        return self._wrapped_scheduler.update_draft_token_ids(draft_token_ids)

    def get_grammar_bitmask(self, scheduler_output):
        """Get grammar bitmask for structured output generation."""
        return self._wrapped_scheduler.get_grammar_bitmask(scheduler_output)
