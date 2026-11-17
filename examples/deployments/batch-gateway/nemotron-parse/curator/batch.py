# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Run a bounded Parse 3.5 single-mode pilot through an external Batch gateway.

The input manifest and rendered pages follow the normal Curator path. This
driver collects a bounded pilot across Curator tasks, submits page requests as
one or more Batch jobs, then uses Curator's existing postprocessor and writer.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import httpx
import pyarrow.parquet as pq
import ray
from batch_client import BatchClient, infer_tasks, page_coverage
from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.interleaved.io import InterleavedParquetWriterStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.inference import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_PATH,
)
from nemo_curator.stages.interleaved.pdf.nemotron_parse.partitioning import (
    PDFPartitioningStage,
)
from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocess import (
    NemotronParsePostprocessStage,
)
from nemo_curator.stages.interleaved.pdf.nemotron_parse.preprocess import (
    PDFPreprocessStage,
)
from pipeline_utils import create_nemotron_parse_pdf_argparser, validate_manifest_path


def main(
    *, client: httpx.Client | None = None, absolute_deadline: float | None = None
) -> None:
    if absolute_deadline is not None and not math.isfinite(absolute_deadline):
        raise ValueError("absolute_deadline must be finite")
    bounded_caller = client is not None or absolute_deadline is not None
    max_upload_file_bytes = 37_425_760 if bounded_caller else None
    max_upload_body_bytes = 37_426_012 if bounded_caller else None

    def check_deadline(stage: str) -> None:
        if absolute_deadline is not None and time.monotonic() >= absolute_deadline:
            raise TimeoutError(f"Original reader deadline expired: {stage}")

    pipeline_profiles: dict[str, dict] = {}

    def run_local_pipeline(pipeline, *, phase: str, **kwargs):
        """Own the local runtime around each canonical Ray pipeline."""
        check_deadline("local Ray startup")
        if ray.is_initialized():
            raise RuntimeError("A fresh caller-owned local Ray runtime is required")
        failure = None
        profile = pipeline_profiles[phase] = {"pipeline_run_completed": False}
        init_started = time.perf_counter()
        try:
            ray.init(
                address="local",
                num_cpus=4,
                include_dashboard=False,
                object_store_memory=512 * 1024**2,
                runtime_env={
                    "env_vars": {
                        "PYTHONPATH": os.environ["PYTHONPATH"],
                        "NO_VCS_VERSION": "1",
                        "CUDA_VISIBLE_DEVICES": "",
                        "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                    }
                },
            )
            profile["ray_init_s"] = time.perf_counter() - init_started
            check_deadline("local Ray startup completion")
            run_started = time.perf_counter()
            try:
                result = pipeline.run(RayDataExecutor(), **kwargs)
                profile["pipeline_run_completed"] = True
                return result
            finally:
                profile["pipeline_run_s"] = time.perf_counter() - run_started
        except BaseException as error:
            failure = error
            raise
        finally:
            if "ray_init_s" not in profile:
                profile["ray_init_s"] = time.perf_counter() - init_started
            shutdown_started = time.perf_counter()
            try:
                ray.shutdown()
            except BaseException as cleanup_error:
                if failure is None:
                    raise
                failure.add_note(
                    "Caller Ray shutdown failed: " + type(cleanup_error).__name__
                )
            finally:
                profile["ray_shutdown_s"] = time.perf_counter() - shutdown_started

    check_deadline("CLI entry")
    parser = create_nemotron_parse_pdf_argparser()
    parser.add_argument(
        "--batch-api-base", required=True, help="Batch gateway API base, including /v1"
    )
    parser.add_argument(
        "--batch-headers-file",
        type=Path,
        help="Private JSON object containing HTTP headers",
    )
    parser.add_argument(
        "--pages-per-job",
        type=int,
        default=64,
        help="Maximum page requests per Batch job",
    )
    args = parser.parse_args()
    if args.parse_mode != "single":
        parser.error("Batch pilot supports only --parse-mode single")
    if (
        args.triple_regions != "union"
        or args.max_tokens != DEFAULT_MAX_TOKENS
        or args.text_in_pic
    ):
        parser.error(
            "Batch pilot does not support --triple-regions, --max-tokens, or --text-in-pic overrides"
        )
    if not args.inference_server_model_name:
        parser.error("--inference-server-model-name is required for the Batch gateway")
    if args.model_path == DEFAULT_MODEL_PATH:
        parser.error(
            "The Batch preview requires the Parse 3.5 model package; Parse 1.2 is unsupported"
        )
    if args.max_pdfs is None or not 0 < args.max_pdfs <= 100:
        parser.error("Set --max-pdfs from 1 to 100 for this bounded in-memory pilot")
    if args.max_pages < 1 or args.max_pdfs * args.max_pages > 1_000:
        parser.error("Limit --max-pdfs times --max-pages to 1,000 rendered pages")
    if not 1 <= args.pages_per_job <= 1_000:
        parser.error("Set --pages-per-job from 1 to 1,000")
    if args.reuse_encoded_images:
        parser.error("The Batch preview does not support --reuse-encoded-images")
    headers = {}
    if args.batch_headers_file is not None:
        headers = json.loads(args.batch_headers_file.read_text())
        if not isinstance(headers, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in headers.items()
        ):
            parser.error(
                "The headers file must contain a JSON object with string values"
            )
    validate_manifest_path(args.manifest)

    args.output_dir = os.path.abspath(args.output_dir)
    os.umask(0o077)
    os.makedirs(args.output_dir, mode=0o700, exist_ok=False)
    started = time.perf_counter()

    prepare = Pipeline(name="nemotron_parse_batch_prepare")
    prepare.add_stage(
        PDFPartitioningStage(
            manifest_path=args.manifest,
            pdfs_per_task=args.pdfs_per_task,
            max_pdfs=args.max_pdfs,
            dataset_name=args.dataset_name,
            file_name_field=args.file_name_field,
            file_names_field=args.file_names_field,
            url_field=args.url_field,
        )
    )
    prepare.add_stage(
        PDFPreprocessStage(
            zip_base_dir=args.zip_base_dir,
            pdf_dir=args.pdf_dir,
            jsonl_base_dir=args.jsonl_base_dir,
            dpi=args.dpi,
            max_pages=args.max_pages,
            retain_failures=True,
        )
    )
    check_deadline("PDF prepare pipeline")
    tasks = run_local_pipeline(prepare, phase="prepare") or []
    check_deadline("PDF prepare pipeline completion")
    if not tasks:
        raise RuntimeError("Batch pilot produced no rendered tasks")
    prepared = time.perf_counter()

    timings: dict = {}
    accounting: dict = {}
    if client is not None and headers:
        raise ValueError(
            "Caller-owned client headers must be bound by its owner; do not also supply a headers file"
        )
    transport_context = (
        nullcontext(client)
        if client is not None
        else httpx.Client(timeout=300.0, headers=headers, follow_redirects=False)
    )
    with transport_context as transport:
        gateway = BatchClient(
            args.batch_api_base,
            client=transport,
            evidence_path=Path(args.output_dir) / "batch_job_events.jsonl",
            absolute_deadline_monotonic=absolute_deadline,
            max_upload_file_bytes=max_upload_file_bytes,
            max_upload_body_bytes=max_upload_body_bytes,
        )
        try:
            inferred = infer_tasks(
                tasks,
                model_name=args.inference_server_model_name,
                model_path=args.model_path,
                gateway=gateway,
                timings=timings,
                accounting=accounting,
                max_lines=args.pages_per_job,
                max_bytes=(
                    max_upload_file_bytes
                    if max_upload_file_bytes is not None
                    else 190_000_000
                ),
            )
        finally:
            accounting.update(
                request_submission_attempts=gateway.request_submission_attempts,
                requests_confirmed_submitted=gateway.requests_confirmed_submitted,
                jobs=gateway.jobs,
                client_profile=gateway.profile,
                timings=timings,
                pipeline_profiles=pipeline_profiles,
                gpu_release_verified=False,
                supervision="Run owner monitors progress, auth and resource cleanup; this client sets no job deadline",
            )
            active_failure = sys.exception()
            try:
                (Path(args.output_dir) / "batch_request_accounting.json").write_text(
                    json.dumps(accounting, indent=2) + "\n"
                )
            except Exception as evidence_error:
                if active_failure is None:
                    raise
                active_failure.add_note(
                    f"Request accounting could not be written: {evidence_error}"
                )
    inferred_at = time.perf_counter()
    check_deadline("inferred output materialization")
    rows = [row for task in inferred for row in task.to_pyarrow().to_pylist()]
    check_deadline("inferred output materialization completion")
    succeeded = sum(bool(row.get("parse_document")) for row in rows)
    errors = sum(bool(row.get("materialize_error")) for row in rows)
    missing = len(rows) - succeeded - errors

    finish = Pipeline(name="nemotron_parse_batch_finish")
    finish.add_stage(NemotronParsePostprocessStage(min_crop_px=args.min_crop_size))
    finish.add_stage(
        InterleavedParquetWriterStage(
            path=args.output_dir,
            materialize_on_write=False,
            on_materialize_error="warn",
        )
    )
    check_deadline("postprocess and write pipeline")
    written = run_local_pipeline(finish, phase="finish", initial_tasks=inferred) or []
    check_deadline("postprocess and write pipeline completion")
    postprocess_pipeline_finished = time.perf_counter()
    output_files = [path for task in written for path in task.data]
    reread_started = time.perf_counter()
    written_rows = 0
    output_error_rows = 0
    output_pages: set[tuple[str, int]] = set()
    for path in output_files:
        check_deadline("Parquet reread")
        table = pq.read_table(
            path, columns=["sample_id", "page_number", "materialize_error"]
        )
        written_rows += table.num_rows
        output_error_rows += sum(
            value is not None for value in table["materialize_error"].to_pylist()
        )
        output_pages.update(
            (row["sample_id"], row["page_number"])
            for row in table.to_pylist()
            if row["page_number"] is not None
        )
    check_deadline("Parquet reread completion")
    reread_finished = time.perf_counter()
    coverage = page_coverage(rows, output_pages)
    finished = time.perf_counter()
    summary = {
        "expected_pages": len(rows),
        "inference_succeeded_pages": succeeded,
        "error_pages": errors,
        "missing_pages": missing,
        "written_files": len(output_files),
        "written_rows": written_rows,
        "output_error_rows": output_error_rows,
        "output_pages": len(output_pages),
        **coverage,
        "client_retries": 0,
        "request_submission_attempts": gateway.request_submission_attempts,
        "requests_confirmed_submitted": gateway.requests_confirmed_submitted,
        "request_accounting_complete": (
            accounting["requests_prepared"]
            == gateway.request_submission_attempts
            == gateway.requests_confirmed_submitted
            == len(accounting["responses"])
            and not accounting["rejected_responses"]
        ),
        "prepare_s": prepared - started,
        "batch_and_replay_s": inferred_at - prepared,
        **timings,
        "postprocess_and_write_s": finished - inferred_at,
        "postprocess_pipeline_and_setup_s": postprocess_pipeline_finished - inferred_at,
        "parquet_reread_s": reread_finished - reread_started,
        "pipeline_profiles": pipeline_profiles,
        "profile_scope": "Caller elapsed time only. Nested fields overlap parent fields. No server queue or GPU inference attribution.",
        "total_s": finished - started,
        "total_clock_boundary": "Before pipeline construction through Parquet reread; excludes process/server startup and summary write",
    }
    check_deadline("summary write")
    (Path(args.output_dir) / "batch_pilot_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    check_deadline("summary write completion")
    print(json.dumps(summary))
    if (
        errors
        or missing
        or len(written) != len(inferred)
        or not written_rows
        or output_error_rows
        or not coverage["structural_page_coverage_complete"]
        or not summary["request_accounting_complete"]
    ):
        raise RuntimeError(
            "Batch pilot incomplete: "
            f"expected={len(rows)} inference_succeeded={succeeded} errors={errors} missing={missing} "
            f"written_tasks={len(written)}/{len(inferred)} written_rows={written_rows} "
            f"output_error_rows={output_error_rows} structural_page_coverage_complete={coverage['structural_page_coverage_complete']}"
        )


if __name__ == "__main__":
    main()
