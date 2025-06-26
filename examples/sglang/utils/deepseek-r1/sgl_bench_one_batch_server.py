"""
This is a modified version of the https://github.com/sgl-project/sglang/blob/main/python/sglang/bench_one_batch_server.py script that allows for benchmarking OpenAI batch completion endpoints. Dynamo has it's own
rust based frontend that supports batch completions so we do not need to use the internal engine `/generate` endpoint.

---
Benchmark the latency of running a single batch with a server.

python3 sgl_bench_one_batch_server.py --model deepseek-ai/DeepSeek-R1 --base-url http://HEAD_PREFILL_NODE_IP:8000 --batch-size 8192 --input-len 4096 --output-len 5 --skip-warmup
python3 sgl_bench_one_batch_server.py --model deepseek-ai/DeepSeek-R1 --base-url http://HEAD_PREFILL_NODE_IP:8000 --batch-size 40000 --input-len 2000 --output-len 100 --skip-warmup
"""

import argparse
import concurrent.futures
import dataclasses
import itertools
import json
import multiprocessing
import os
import random
import threading
import time
from typing import Tuple

import requests
from sglang.bench_serving import get_tokenizer, sample_random_requests
from sglang.profiler import run_profile
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import is_in_ci, write_github_step_summary


@dataclasses.dataclass
class BenchArgs:
    run_name: str = "default"
    batch_size: Tuple[int] = (1,)
    input_len: Tuple[int] = (1024,)
    output_len: Tuple[int] = (16,)
    temperature: float = 0.0
    return_logprob: bool = False
    input_len_step_percentage: float = 0.0
    result_filename: str = "result.jsonl"
    base_url: str = ""
    skip_warmup: bool = False
    show_report: bool = False
    profile: bool = False
    profile_by_stage: bool = False

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--run-name", type=str, default=BenchArgs.run_name)
        parser.add_argument(
            "--batch-size", type=int, nargs="+", default=BenchArgs.batch_size
        )
        parser.add_argument(
            "--input-len", type=int, nargs="+", default=BenchArgs.input_len
        )
        parser.add_argument(
            "--output-len", type=int, nargs="+", default=BenchArgs.output_len
        )
        parser.add_argument("--temperature", type=float, default=BenchArgs.temperature)
        parser.add_argument("--return-logprob", action="store_true")
        parser.add_argument(
            "--input-len-step-percentage",
            type=float,
            default=BenchArgs.input_len_step_percentage,
        )
        parser.add_argument(
            "--result-filename", type=str, default=BenchArgs.result_filename
        )
        parser.add_argument("--base-url", type=str, default=BenchArgs.base_url)
        parser.add_argument("--skip-warmup", action="store_true")
        parser.add_argument("--show-report", action="store_true")
        parser.add_argument("--profile", action="store_true")
        parser.add_argument("--profile-by-stage", action="store_true")

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # use the default value's type to cast the args into correct types.
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        return cls(
            **{attr: attr_type(getattr(args, attr)) for attr, attr_type in attrs}
        )


def launch_server_internal(server_args):
    try:
        launch_server(server_args)
    except Exception as e:
        raise e
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


def launch_server_process(server_args: ServerArgs):
    proc = multiprocessing.Process(target=launch_server_internal, args=(server_args,))
    proc.start()
    base_url = f"http://{server_args.host}:{server_args.port}"
    timeout = 600

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            headers = {
                "Content-Type": "application/json; charset=utf-8",
            }
            response = requests.get(f"{base_url}/v1/models", headers=headers)
            if response.status_code == 200:
                return proc, base_url
        except requests.RequestException:
            pass
        time.sleep(10)
    raise TimeoutError("Server failed to start within the timeout period.")


def estimate_payload_size(batch_size: int, input_len: int) -> int:
    """Estimate JSON payload size in bytes"""
    # Each token ID: ~4 bytes average when JSON serialized
    # Plus JSON structure overhead (~20% additional)
    token_bytes = batch_size * input_len * 4
    overhead = token_bytes * 0.2
    return int(token_bytes + overhead)


def chunk_batch(input_requests, max_payload_bytes: int = 64 * 1024 * 1024):  # 64MB
    """Split large batch into smaller chunks that fit NATS limit"""
    if not input_requests:
        return []

    # Estimate size of one request
    sample_size = len(json.dumps({"prompt": input_requests[0].prompt}))
    requests_per_chunk = max(1, max_payload_bytes // sample_size)

    chunks = []
    for i in range(0, len(input_requests), requests_per_chunk):
        chunk = input_requests[i : i + requests_per_chunk]
        chunks.append(chunk)

    print(
        f"Split batch of {len(input_requests)} into {len(chunks)} chunks of ~{requests_per_chunk} each"
    )
    return chunks


def run_one_case(
    url: str,
    batch_size: int,
    input_len: int,
    output_len: int,
    temperature: float,
    return_logprob: bool,
    input_len_step_percentage: float,
    run_name: str,
    result_filename: str,
    tokenizer,
    profile: bool = False,
    profile_by_stage: bool = False,
):
    input_requests = sample_random_requests(
        input_len=input_len,
        output_len=output_len,
        num_prompts=batch_size,
        range_ratio=1.0,
        tokenizer=tokenizer,
        dataset_path="",
        random_sample=True,
        return_text=False,
    )

    # In Dynamo, we use the async-openai library which has sa u16 max size limit for integer based prompt representations
    # We have a PR to bump this to u32 to handle a larger token size https://github.com/64bit/async-openai/pull/392
    # For now we swap tokens that are larger with a random number between 0 and U16_MAX_SIZE
    U16_MAX_SIZE = 50256
    for req in input_requests:
        if hasattr(req, "prompt") and isinstance(req.prompt, list):
            req.prompt = [
                random.randint(0, U16_MAX_SIZE) if token > U16_MAX_SIZE else token
                for token in req.prompt
            ]

    # In order to work around the maximum NATS message size, we have to chunk the batch and then send
    estimated_size = estimate_payload_size(batch_size, input_len)
    print(f"Estimated payload size: {estimated_size / (1024*1024):.1f}MB")

    if (
        estimated_size > 64 * 1024 * 1024
    ):  # nats max payload size is 64MB but recommended size is 8MB
        chunks = chunk_batch(input_requests)
        return run_chunked_requests(url, chunks, batch_size, input_len, output_len, 
                                  temperature, return_logprob, input_len_step_percentage,
                                  run_name, result_filename, tokenizer, profile, profile_by_stage)
    
    profile_link = None
    if profile:
        profile_link: str = run_profile(
            url, 3, ["CPU", "GPU"], None, None, profile_by_stage
        )

    tic = time.perf_counter()
    request_payload = {
        "model": "deepseek-ai/DeepSeek-R1",
        "prompt": [req.prompt for req in input_requests],
        "stream": True,
    }

    print("==== Sending request to", url + "/v1/completions")

    response = requests.post(url + "/v1/completions", json=request_payload, stream=True)

    ttft = 0.0

    for chunk in response.iter_lines(decode_unicode=False):
        chunk = chunk.decode("utf-8")
        if chunk.startswith("data: [DONE]"):
            break
        if not chunk.startswith("data:"):
            continue

        data_str = chunk[len("data:") :].strip()
        try:
            data = json.loads(data_str)
        except Exception:
            print("[WARN] Failed to parse chunk:", chunk)
            continue

        # OpenAI-style structure
        usage = data.get("usage")
        if usage and usage.get("completion_tokens") == 1:
            ttft = time.perf_counter() - tic

    latency = time.perf_counter() - tic
    input_throughput = batch_size * input_len / ttft
    output_throughput = batch_size * output_len / (latency - ttft)
    overall_throughput = batch_size * (input_len + output_len) / latency

    print(f"batch size: {batch_size}")
    print(f"input_len: {input_len}")
    print(f"output_len: {output_len}")
    print(f"latency: {latency:.2f} s")
    print(f"ttft: {ttft:.2f} s")
    # print(f"last generation throughput: {last_gen_throughput:.2f} tok/s")
    print(f"input throughput: {input_throughput:.2f} tok/s")
    if output_len != 1:
        print(f"output throughput: {output_throughput:.2f} tok/s")

    if result_filename:
        with open(result_filename, "a") as fout:
            res = {
                "run_name": run_name,
                "batch_size": batch_size,
                "input_len": input_len,
                "output_len": output_len,
                "ttft": round(ttft, 4),
                "latency": round(latency, 4),
                "output_throughput": round(output_throughput, 2),
                "overall_throughput": round(overall_throughput, 2),
                # "last_gen_throughput": round(last_gen_throughput, 2),
            }
            fout.write(json.dumps(res) + "\n")

    return (
        batch_size,
        latency,
        ttft,
        input_throughput,
        output_throughput,
        overall_throughput,
        profile_link if profile else None,
    )


def run_chunked_requests(
    url,
    chunks,
    original_batch_size,
    input_len,
    output_len,
    temperature,
    return_logprob,
    input_len_step_percentage,
    run_name,
    result_filename,
    tokenizer,
    profile,
    profile_by_stage,
):
    """Process large batch as multiple parallel requests"""

    print(
        f"Processing {len(chunks)} chunks for total batch size {original_batch_size} IN PARALLEL"
    )

    total_start = time.perf_counter()
    first_token_time = None
    first_token_lock = threading.Lock()

    def process_chunk(chunk_info):
        i, chunk = chunk_info
        print(f"Starting chunk {i+1}/{len(chunks)} with {len(chunk)} requests")

        chunk_start = time.perf_counter()

        request_payload = {
            "model": "deepseek-ai/DeepSeek-R1",
            "prompt": [req.prompt for req in chunk],
            "stream": True,
        }

        response = requests.post(
            url + "/v1/completions", json=request_payload, stream=True
        )
        response.raise_for_status()

        chunk_first_token = None
        for line in response.iter_lines(decode_unicode=False):
            line = line.decode("utf-8")
            if line.startswith("data: [DONE]"):
                break
            if not line.startswith("data:"):
                continue

            data_str = line[len("data:") :].strip()
            try:
                data = json.loads(data_str)
            except:
                continue

            usage = data.get("usage")
            if usage and usage.get("completion_tokens") == 1:
                chunk_first_token = time.perf_counter() - chunk_start

                # Thread-safe first token tracking
                nonlocal first_token_time
                with first_token_lock:
                    if first_token_time is None:
                        first_token_time = time.perf_counter() - total_start
                break

        chunk_end = time.perf_counter()
        print(f"Completed chunk {i+1}/{len(chunks)} in {chunk_end - chunk_start:.2f}s")

        return {
            "latency": chunk_end - chunk_start,
            "first_token": chunk_first_token,
            "size": len(chunk),
            "chunk_id": i,
        }

    # Process all chunks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        chunk_results = list(executor.map(process_chunk, enumerate(chunks)))

    total_latency = time.perf_counter() - total_start
    ttft = first_token_time or 0.0

    # Calculate aggregated metrics
    input_throughput = original_batch_size * input_len / ttft if ttft > 0 else 0
    output_throughput = (
        original_batch_size * output_len / (total_latency - ttft)
        if total_latency > ttft
        else 0
    )
    overall_throughput = original_batch_size * (input_len + output_len) / total_latency

    print("Chunked batch results:")
    print(f"  Total latency: {total_latency:.2f}s")
    print(f"  TTFT: {ttft:.2f}s")
    print(f"  Input throughput: {input_throughput:.2f} tok/s")
    print(f"  Output throughput: {output_throughput:.2f} tok/s")

    # Save results
    if result_filename:
        with open(result_filename, "a") as fout:
            res = {
                "run_name": run_name,
                "batch_size": original_batch_size,
                "input_len": input_len,
                "output_len": output_len,
                "ttft": round(ttft, 4),
                "latency": round(total_latency, 4),
                "output_throughput": round(output_throughput, 2),
                "overall_throughput": round(overall_throughput, 2),
                "chunks": len(chunks),
            }
            fout.write(json.dumps(res) + "\n")

    return (
        original_batch_size,
        total_latency,
        ttft,
        input_throughput,
        output_throughput,
        overall_throughput,
        None,
    )


def run_benchmark(server_args: ServerArgs, bench_args: BenchArgs):
    if bench_args.base_url:
        proc, base_url = None, bench_args.base_url
    else:
        proc, base_url = launch_server_process(server_args)

    tokenizer_id = server_args.tokenizer_path or server_args.model_path
    tokenizer = get_tokenizer(tokenizer_id)

    # warmup
    if not bench_args.skip_warmup:
        print("=" * 8 + " Warmup Begin " + "=" * 8)
        run_one_case(
            base_url,
            batch_size=16,
            input_len=1024,
            output_len=16,
            temperature=bench_args.temperature,
            return_logprob=bench_args.return_logprob,
            input_len_step_percentage=bench_args.input_len_step_percentage,
            run_name="",
            result_filename="",
            tokenizer=tokenizer,
        )
        print("=" * 8 + " Warmup End   " + "=" * 8 + "\n")

    # benchmark
    result = []
    bench_result = []
    try:
        for bs, il, ol in itertools.product(
            bench_args.batch_size, bench_args.input_len, bench_args.output_len
        ):
            result.append(
                run_one_case(
                    base_url,
                    bs,
                    il,
                    ol,
                    temperature=bench_args.temperature,
                    return_logprob=bench_args.return_logprob,
                    input_len_step_percentage=bench_args.input_len_step_percentage,
                    run_name=bench_args.run_name,
                    result_filename=bench_args.result_filename,
                    tokenizer=tokenizer,
                )
            )

        if bench_args.profile:
            try:
                for bs, il, ol in itertools.product(
                    bench_args.batch_size, bench_args.input_len, bench_args.output_len
                ):
                    bench_result.append(
                        (
                            run_one_case(
                                base_url,
                                bs,
                                il,
                                ol,
                                temperature=bench_args.temperature,
                                return_logprob=bench_args.return_logprob,
                                input_len_step_percentage=bench_args.input_len_step_percentage,
                                run_name=bench_args.run_name,
                                result_filename=bench_args.result_filename,
                                tokenizer=tokenizer,
                                profile=bench_args.profile,
                                profile_by_stage=bench_args.profile_by_stage,
                            )[-1],
                        )
                    )
                result = [t1[:-1] + t2 for t1, t2 in zip(result, bench_result)]
            except Exception as e:
                print(f"Error profiling, there will be no profile trace dump: {e}")
    finally:
        if proc:
            kill_process_tree(proc.pid)

    print(f"\nResults are saved to {bench_args.result_filename}")

    if not bench_args.show_report:
        return

<<<<<<< HEAD
    summary = (
        f"\nInput length: {bench_args.input_len}. Output length: {bench_args.output_len}.\n"
    )
=======
    summary = f"\nInput lenses: {bench_args.input_len}. Output lenses: {bench_args.output_len}.\n"
>>>>>>> 5253631fcc7e3d6d0d6e3764923d6239e2d36e96
    summary += "| batch size | latency (s) | input throughput (tok/s)  | output throughput (tok/s) | acc length | ITL (ms) | input cost ($/1M) | output cost ($/1M) |"

    if bench_args.profile:
        summary += " profile |"

    summary += "\n"
    summary += "| ---------- | ----------- | ------------------------- | ------------------------- | ---------- | -------- | ----------------- | ------------------ |"

    if bench_args.profile:
        summary += "-------------|"
    summary += "\n"

    for (
        batch_size,
        latency,
        ttft,
        input_throughput,
        output_throughput,
        overall_throughput,
        last_gen_throughput,
        acc_length,
        trace_link,
    ) in result:
        hourly_cost = 2 * server_args.tp_size  # $2/hour for one H100
        input_util = 0.7
        accept_length = round(acc_length, 2) if acc_length is not None else "n/a"
        line = (
            f"| {batch_size} | "
            f"{latency:.2f} | "
            f"{input_throughput:.2f} | "
            f"{output_throughput:.2f} | "
            f"{accept_length} | "
            f"{1 / (output_throughput/batch_size) * 1000:.2f} | "
            f"{1e6 / (input_throughput * input_util) / 3600 * hourly_cost:.2f} | "
            f"{1e6 / output_throughput / 3600 * hourly_cost:.2f} |"
        )
        if trace_link:
            line += f" [Profile]({trace_link}) |"
        line += "\n"
        summary += line

    # print metrics table
    print(summary)

    if is_in_ci():
        write_github_step_summary(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    run_benchmark(server_args, bench_args)
