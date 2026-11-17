<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Validation and profiling notes

Use this page to check what has been tested and to compare a new run with the recorded experiments. Start with [the recipe](README.md) to run your own PDF.

## Tested workflow

An earlier one-page host test prepared a real PDF, submitted it through the public Files and Batch APIs, downloaded the response, and wrote and reread 13 Parquet rows. The expected title, body and contact text were present. The test used synchronous dispatch and a restricted HTTPX transport.

The current client also rejects decoded documents with no blocks and records phase timings. These changes have offline coverage. A live run of this revision, a clean installation, ordinary HTTPX networking and the included Kubernetes Async profile still need testing. The one-page result does not establish OCR accuracy for other documents or a performance advantage.

The tutorial targets Curator commit `b56bd6cd4fd563d2acadc463cbf01c92a12ef0bb`. It uses Parse 3.5 revision `62cb1e4ba153d113ff64aaf2be6e068c377cf53c`. The recorded client environment used Python 3.12, Ray 2.57.0, HTTPX 0.28.1, PyArrow 23.0.1 and CPU PyTorch 2.11.0+cpu. These are recorded versions, not an installation lock.

## Host resource settings

The host test gave the reader four CPUs, 16 GiB of memory and a 1,024 PID limit. Its `/work` directory was a 7 GiB tmpfs. `/tmp` and shared memory were each 512 MiB. All three allocations were inside the same 16 GiB reader limit. The Parse model ran separately on the endpoint's GPU. This layout was specific to that experiment. The current source lets Ray choose its default temporary directory.

The tutorial starts a fresh local Ray runtime for PDF preparation and another for postprocessing. Each runtime uses four Ray CPUs and a 512 MiB object store. The executor and caller shut down the local runtime after use, including handled failures. The tutorial does not connect to an existing Ray cluster.

## Input and request limits

The adapter supports `single` Parse mode. It accepts at most 100 PDFs and 1,000 rendered pages per invocation, with at most 300,000,000 rendered-image bytes. It retains the images in memory through replay. A small page count can still exceed the byte limit.

The recipe's Python caller supplies a 600-second client deadline and upload limits of 37,425,760 bytes for JSONL and 37,426,012 bytes for multipart data. A complex page can exceed those limits. Running `batch.py` directly omits the optional overall deadline and these upload caps. When calling `infer_tasks` directly, pass a `BatchClient` with an explicit deadline and supervise the process externally.

The deadline is checked by the client. It does not terminate the process or remote inference. Cancellation is attempted after a handled error only while time remains. The endpoint operator must reconcile accepted jobs and check remote cleanup after an error or disconnect.

## Larger-run findings

The earlier cached-input comparison used 512 pages in both arms. Direct took 240.130 seconds and synchronous Batch took 260.389 seconds. Each arm returned six token-limit failures. The remaining pages were not independently checked for complete OCR accuracy.

A later Direct attempt admitted all 512 pages and again returned six token-limit failures. Its Parquet file exceeded the experiment's 64 MiB per-file acceptance limit, and a separate output-validation check failed. Batch did not run. This attempt cannot support a paired timing comparison, and the artifact limit alone does not establish a product limitation.

A six-page diagnostic then tested model-card generation settings in both Direct and Batch. Each mode returned two parseable candidates and four token-limit failures. Important fields or table cells were missing from the candidates. Generation settings alone did not resolve the quality problem. No page was dropped from the counts.

The earlier one-page host test did not retain a rendered PNG checksum or establish visual grounding. Quality checks should compare the output with the source PDF, including tables, images and apparently blank pages.

## Deployment template

The [Parse profile](profile-parse.yaml) configures the existing Dynamo Batch chart for an operator-provided Parse deployment. Set its DGD, model and storage values before rendering. It creates no model workers. Component versions come from the parent chart, so check those pins when reproducing a run.

The passing host test used synchronous dispatch. Async deployment, customer authentication, disconnect recovery, online traffic coexistence and repeated performance measurements require their own checks.

## Read the caller profile

The client records elapsed timings for new runs. The historical 8.44% Parse gap was measured with an earlier client, so these fields cannot explain that result retrospectively.

`client_profile.phases` contains cumulative `elapsed_s`, `calls` and `failures` for each phase. Counters persist across jobs and repeated `BatchClient.run` calls on the same client. They include failed operations. `runs_started`, `runs_completed` and `runs_failed` describe transport reconciliation, not Parse content quality. A later successful run does not erase an earlier failure. A run still in progress is started but neither completed nor failed.

| Phase | Client clock boundary |
| --- | --- |
| `upload` | Upload preparation, multipart construction, HTTP exchange and upload response validation |
| `multipart_encode` | Explicit multipart construction in the bounded caller path, nested inside `upload`. Absent in the ordinary path |
| `job_create` | Create request through response identity validation |
| `status_http` | Each status HTTP exchange through response identity validation. Excludes time between polls |
| `result_download` | Each result or error file HTTP exchange through HTTP status validation. Excludes JSONL reconciliation and Parse replay |
| `poll_sleep` | Actual elapsed time spent in the client's poll delay, including interrupted delays |

For example, a large `upload.elapsed_s` makes upload preparation and transport worth inspecting. A large `poll_sleep.elapsed_s` shows how long the client slept while jobs remained pending. It cannot distinguish server queueing, inference, output persistence or polling observation delay. Server traces are required for that breakdown.

The summary retains `prepare_s`, `batch_and_replay_s`, `request_build_s`, `gateway_s`, `replay_s`, `postprocess_and_write_s` and `total_s`. `gateway_s` still includes JSONL construction, job submission, polling, downloads and reconciliation. `postprocess_and_write_s` keeps its earlier broad boundary through Parquet reread and coverage accounting. New `postprocess_pipeline_and_setup_s` ends after the finish pipeline and its local Ray shutdown. `parquet_reread_s` measures output file iteration, rereading and row accounting before coverage evaluation.

`pipeline_profiles.prepare` and `pipeline_profiles.finish` split each caller-owned local runtime into `ray_init_s`, `pipeline_run_s` and `ray_shutdown_s`. `pipeline_run_completed` only describes that pipeline call. Curator stages may overlap inside `pipeline_run_s`, and the executor may perform runtime work there. These fields do not measure individual worker stages.

Nested fields overlap their parents. Do not add them into a second total or label any field as GPU inference time. The total starts before pipeline construction and ends after Parquet reread and coverage accounting. It excludes client process startup, server startup and summary writing. A failed run may leave only partial request accounting and events, with no summary. Events are ordinary append writes, not a crash-durable tracing system. New timing events are best effort and do not replace transport errors.
