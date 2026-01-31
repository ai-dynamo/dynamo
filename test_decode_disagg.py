"""Test script for decode disaggregation with interactive UI."""

import argparse
import json
import os
import random
import select
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import torch
from transformers import AutoTokenizer

TIER_BOUNDARIES = [512, 1024, 4096]  # Token boundaries for decode tiers
HEALTH_URL = "http://localhost:8080/health"
CHAT_URL = "http://localhost:8080/v1/chat/completions"
DEFAULT_DOCKER_IMAGE = "nvcr.io/nvidian/dynamo-dev/warnold-utils:sglang-dd-058-v3-amd64"

# Event to signal all processes should stop
stop_event = threading.Event()

# Global config for docker mode
docker_config: dict = {}

# Global config for claude mode (logs to files)
claude_mode = False
log_files: dict[str, any] = {}

# Track processes with names for monitoring
processes: dict[str, subprocess.Popen] = {}

# ANSI colors for log prefixes
COLORS = {
    "prefill": "\033[32m",  # green
    "decode1": "\033[36m",  # cyan
    "decode2": "\033[35m",  # magenta
    "decode3": "\033[34m",  # blue
    "frontend": "\033[33m",  # yellow
    "reset": "\033[0m",
}


def get_num_gpus() -> int:
    """Detect the number of available GPUs."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def get_common_args(model_path: Path, mem_fraction: float) -> list[str]:
    """Get common arguments with specified mem-fraction-static."""
    max_context_length = max(TIER_BOUNDARIES)
    return [
        "--model-path",
        str(model_path),
        "--served-model-name",
        "model",
        "--context-length",
        str(max_context_length),
        "--max-total-tokens",
        str(5 * max_context_length),
        "--mem-fraction-static",
        str(mem_fraction),
        "--page-size",
        "16",
        "--disable-cuda-graph",
        "--stream-interval",
        "100",
        "--log-level",
        "debug",
    ]


# Load tokenizer for boundary analysis
tokenizer = None
_tokenizer_model_path = None


def get_tokenizer(model_path: Path):
    """Lazily load the tokenizer."""
    global tokenizer, _tokenizer_model_path
    if tokenizer is None or _tokenizer_model_path != model_path:
        print(f"📚 Loading tokenizer from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        _tokenizer_model_path = model_path
    return tokenizer


def format_with_boundaries(text: str, prompt: str, model_path: Path) -> str:
    """Format text with markers at tier boundaries.

    Returns the text with colored markers showing where token boundaries are.
    """
    tok = get_tokenizer(model_path)

    # Tokenize prompt to get its length
    prompt_tokens = tok.encode(prompt)
    prompt_len = len(prompt_tokens)

    # Tokenize full output
    output_tokens = tok.encode(text, add_special_tokens=False)

    # Build output with boundary markers
    result_parts = []
    current_pos = 0

    for boundary in TIER_BOUNDARIES:
        # Boundary is relative to total tokens (prompt + output)
        output_boundary = boundary - prompt_len

        if output_boundary <= 0:
            # Boundary is within prompt, mark at start
            if current_pos == 0:
                result_parts.append(f"\033[41m[←{boundary}]\033[0m")
        elif output_boundary < len(output_tokens):
            # Decode tokens up to boundary
            tokens_before = output_tokens[current_pos:output_boundary]
            text_before = tok.decode(tokens_before)
            result_parts.append(text_before)
            result_parts.append(f"\033[41m[{boundary}]\033[0m")
            current_pos = output_boundary

    # Add remaining tokens
    if current_pos < len(output_tokens):
        remaining_tokens = output_tokens[current_pos:]
        result_parts.append(tok.decode(remaining_tokens))

    formatted = "".join(result_parts)

    # Add summary
    total_tokens = prompt_len + len(output_tokens)
    summary = f"\n\n📊 Token counts: prompt={prompt_len}, output={len(output_tokens)}, total={total_tokens}"
    for boundary in TIER_BOUNDARIES:
        if total_tokens >= boundary:
            summary += f"\n   ✓ Crossed {boundary} boundary at output token {boundary - prompt_len}"

    return formatted + summary


def stream_output(proc: subprocess.Popen, name: str, stream: str):
    """Stream stdout/stderr from a process with a colored prefix or to file."""
    color = COLORS.get(name, "")
    reset = COLORS["reset"]
    prefix = f"{color}[{name}]{reset} "

    pipe = proc.stdout if stream == "stdout" else proc.stderr

    # In claude mode, write to file instead of stdout
    if claude_mode:
        log_file = log_files.get(name)
        for line in iter(pipe.readline, ""):
            if line and log_file:
                log_file.write(f"[{stream}] {line}")
                log_file.flush()
    else:
        for line in iter(pipe.readline, ""):
            if line:
                print(f"{prefix}{line}", end="", flush=True)


def start_worker(
    name: str, args: list[str], env: dict[str, str] | None = None
) -> subprocess.Popen:
    """Start a worker process and stream its logs."""
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)

    # Wrap in docker run if docker mode is enabled
    if docker_config.get("enabled"):
        args = build_docker_command(name, args, env)
        # Don't pass env to Popen when using docker (it's passed via -e flags)
        proc_env = os.environ.copy()

    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=proc_env,
    )

    # Register process for monitoring
    processes[name] = proc

    # Start threads to stream stdout and stderr
    threading.Thread(
        target=stream_output, args=(proc, name, "stdout"), daemon=True
    ).start()
    threading.Thread(
        target=stream_output, args=(proc, name, "stderr"), daemon=True
    ).start()

    return proc


def cleanup_docker_container(name: str):
    """Remove existing docker container if it exists."""
    container_name = f"dynamo-{name}"
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
    )


def build_docker_command(
    name: str, args: list[str], env: dict[str, str] | None = None
) -> list[str]:
    """Wrap a command in docker run."""
    # Remove any existing container with the same name
    cleanup_docker_container(name)

    image = docker_config["image"]
    # Resolved path (symlinks don't exist inside container)
    resolved_model_path = Path(docker_config["model_path"])
    resolved_models_dir = resolved_model_path.parent

    # Original path (with symlinks) for matching in args
    original_model_path = docker_config["model_path_original"]
    original_models_dir = str(Path(original_model_path).parent)

    # Get GPU set from env if specified
    gpu_devices = env.get("CUDA_VISIBLE_DEVICES") if env else None

    # Get absolute paths to local source code
    local_components_src = (Path(__file__).parent / "components" / "src").resolve()
    local_bindings_src = (
        Path(__file__).parent / "lib" / "bindings" / "python" / "src"
    ).resolve()
    local_sglang_src = (Path.home() / "proj" / "sglang" / "python" / "sglang").resolve()

    docker_args = [
        "docker",
        "run",
        "--rm",
        "-i",
        "--name",
        f"dynamo-{name}",
        "--gpus",
        "all",
        "--network",
        "host",
        "--ipc",
        "host",
        "--shm-size",
        "16g",
        "--ulimit",
        "memlock=-1",
        # Mount models directory (resolved path)
        "-v",
        f"{resolved_models_dir}:{resolved_models_dir}",
        # Mount local source code to workspace directories
        "-v",
        f"{local_components_src}:/workspace/components/src",
        "-v",
        f"{local_bindings_src}:/workspace/lib/bindings/python/src",
        "-v",
        f"{local_sglang_src}:/workspace/sglang",
        # Set PYTHONPATH to prioritize our local code
        "-e",
        "PYTHONPATH=/workspace/components/src:/workspace/lib/bindings/python/src:/workspace",
    ]

    # Set CUDA_VISIBLE_DEVICES inside container for GPU selection
    if gpu_devices:
        docker_args.extend(["-e", f"CUDA_VISIBLE_DEVICES={gpu_devices}"])

    # Add other environment variables
    if env:
        for key, value in env.items():
            if key == "CUDA_VISIBLE_DEVICES":
                continue  # Already handled above
            docker_args.extend(["-e", f"{key}={value}"])

    # Add the image and command
    docker_args.append(image)

    # Convert the original args: replace sys.executable with python3
    # and rewrite any paths containing the original (symlink) model dir to resolved path
    converted_args = []
    for arg in args:
        if arg == sys.executable:
            converted_args.append("python3")
        elif original_models_dir in arg:
            # Rewrite symlink path to resolved path
            converted_args.append(
                arg.replace(original_models_dir, str(resolved_models_dir))
            )
        else:
            converted_args.append(arg)

    docker_args.extend(converted_args)

    return docker_args


def monitor_processes():
    """Monitor all processes and signal stop if any exits."""
    while not stop_event.is_set():
        for name, proc in list(processes.items()):
            ret = proc.poll()
            if ret is not None:
                color = COLORS.get(name, "")
                reset = COLORS["reset"]
                print(f"\n{color}💀 [{name}] exited with code {ret}{reset}")
                stop_event.set()
                return
        time.sleep(0.5)


def wait_for_health(num_workers: int, timeout: float = 600.0):
    """Poll /health until we have the expected number of instances."""
    print(f"\n⏳ Waiting for {num_workers} workers to be ready...")
    start = time.time()

    while time.time() - start < timeout:
        if stop_event.is_set():
            print("\n❌ A process exited, aborting health check")
            return False

        try:
            resp = httpx.get(HEALTH_URL, timeout=2.0)
            data = resp.json()
            instances = data.get("instances", [])
            count = len(set([a["instance_id"] for a in instances]))

            if count >= num_workers:
                print(f"✅ All {num_workers} workers are ready!")
                return True
            else:
                print(f"   ... {count}/{num_workers} workers ready", end="\r")
        except Exception:
            print("   ... waiting for frontend to start", end="\r")

        time.sleep(1.0)

    print(f"\n❌ Timeout waiting for workers after {timeout}s")
    return False


def random_prompt() -> str:
    """Generate a random prompt."""
    topics = [
        "Tell me a story about",
        "Explain how to",
        "Write a poem about",
        "Describe the process of",
        "What are the benefits of",
        "Give me tips for",
    ]
    subjects = [
        "a brave knight",
        "cooking pasta",
        "the ocean",
        "learning a new language",
        "meditation",
        "space exploration",
        "ancient civilizations",
        "machine learning",
    ]
    return f"{random.choice(topics)} {random.choice(subjects)}"


def run_request(max_tokens: int, model_path: Path):
    """Run a chat completion request with streaming."""
    prompt = random_prompt()
    print(f"\n📝 Prompt: {prompt}", flush=True)
    print(f"🎯 Max tokens: {max_tokens}", flush=True)
    print("-" * 50, flush=True)

    payload = {
        "model": "model",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "ignore_eos": True,  # Always generate exactly max_tokens for debugging
    }

    try:
        full_output = []
        print("⏳ Generating... (output will be shown at the end)", flush=True)
        print(f"🔗 Sending request to {CHAT_URL}...", flush=True)
        with httpx.stream("POST", CHAT_URL, json=payload, timeout=120.0) as resp:
            print(f"✓ Connected, status: {resp.status_code}", flush=True)
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            full_output.append(content)
                    except json.JSONDecodeError:
                        pass
        print("-" * 50)
        final_text = "".join(full_output)
        print(f"\n📋 Final output ({len(final_text)} chars):")

        # Format with tier boundary markers
        formatted_output = format_with_boundaries(final_text, prompt, model_path)
        print(formatted_output)
        print("-" * 50)
    except Exception as e:
        print(f"\n❌ Request failed: {e}")


def run_automated_requests():
    """Run a few concurrent curl requests for Claude mode."""

    print("\n" + "=" * 50)
    print("🤖 Claude Mode - Running Automated Requests")
    print("=" * 50 + "\n")

    # Test with different OSL values to trigger migrations
    test_osls = [100, 600, 1200, 3000]  # Small, tier1, tier2, tier3

    def run_single_request(osl: int):
        """Run a single curl request."""
        payload = {
            "model": "model",
            "messages": [{"role": "user", "content": f"Count to {osl}"}],
            "max_tokens": osl,
            "stream": False,
            "ignore_eos": True,
        }

        print(f"🚀 Starting request with OSL={osl}")
        try:
            resp = httpx.post(CHAT_URL, json=payload, timeout=120.0)
            if resp.status_code == 200:
                result = resp.json()
                completion_tokens = result.get("usage", {}).get("completion_tokens", 0)
                print(f"✅ OSL={osl} completed: {completion_tokens} tokens generated")
            else:
                print(f"❌ OSL={osl} failed with status {resp.status_code}")
        except Exception as e:
            print(f"❌ OSL={osl} failed: {e}")

    # Run requests sequentially to see migrations clearly
    print(f"Running {len(test_osls)} requests sequentially...\n")
    for osl in test_osls:
        run_single_request(osl)
        time.sleep(2)  # Small delay between requests

    print("\n" + "=" * 50)
    print("✅ All automated requests completed")
    print("=" * 50)


def interactive_loop(model_path: Path):
    """Interactive loop for entering OSL and running requests."""
    print("\n" + "=" * 50)
    print("🎮 Interactive Mode")
    print("Enter an output sequence length (OSL) to run a request.")
    print("Type 'q' or 'quit' to exit.")
    print("=" * 50 + "\n")

    while not stop_event.is_set():
        try:
            # Show prompt
            print("OSL> ", end="", flush=True)

            # Use a polling approach so we can check stop_event
            while not stop_event.is_set():
                if select.select([sys.stdin], [], [], 0.5)[0]:
                    user_input = sys.stdin.readline().strip()
                    break
            else:
                print("\n⚠️  A process exited, stopping...")
                break

            if user_input.lower() in ("q", "quit", "exit"):
                print("👋 Goodbye!")
                break

            if not user_input:
                continue

            try:
                osl = int(user_input)
                if osl <= 0:
                    print("⚠️  Please enter a positive integer.")
                    continue
                run_request(osl, model_path)
            except ValueError:
                print("⚠️  Please enter a valid integer.")
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

    if stop_event.is_set():
        print("\n⚠️  Exiting due to process failure")


def cleanup():
    """Terminate all running processes."""
    print("\n🧹 Cleaning up processes...")

    # If docker mode, stop containers by name
    if docker_config.get("enabled"):
        for name in processes.keys():
            container_name = f"dynamo-{name}"
            subprocess.run(
                ["docker", "stop", container_name],
                capture_output=True,
                timeout=10,
            )

    for name, proc in processes.items():
        if proc.poll() is None:  # Still running
            proc.terminate()
    for name, proc in processes.items():
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("✅ Done!")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Test script for decode disaggregation with interactive UI"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(Path.home() / "proj/models/qwen3-4b"),
        help="Path to the model directory (default: ~/proj/models/smol2-135m)",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Run workers inside Docker containers",
    )
    parser.add_argument(
        "--docker-image",
        type=str,
        default=DEFAULT_DOCKER_IMAGE,
        help=f"Docker image to use (default: {DEFAULT_DOCKER_IMAGE})",
    )
    parser.add_argument(
        "--claude",
        action="store_true",
        help="Claude mode: log to files and run automated concurrent requests",
    )
    args = parser.parse_args()

    # Configure docker mode
    if args.docker:
        docker_config["enabled"] = True
        docker_config["image"] = args.docker_image
        docker_config[
            "model_path_original"
        ] = args.model_path  # Keep original (with symlinks)
        docker_config["model_path"] = str(
            Path(args.model_path).resolve()
        )  # Resolved path
        print(f"🐳 Docker mode enabled with image: {args.docker_image}")

        # Pull the image once before starting workers
        print(f"📥 Pulling Docker image: {args.docker_image}")
        try:
            result = subprocess.run(
                ["docker", "pull", args.docker_image],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                print(f"❌ Failed to pull Docker image: {result.stderr}")
                return 1
            print("✅ Docker image pulled successfully")
        except subprocess.TimeoutExpired:
            print("❌ Docker pull timed out after 600 seconds")
            return 1
        except Exception as e:
            print(f"❌ Failed to pull Docker image: {e}")
            return 1

    # Configure claude mode
    global claude_mode, log_files
    if args.claude:
        claude_mode = True
        print("🤖 Claude mode enabled: logging to /tmp files")
        # Create log files for each worker
        for worker_name in ["prefill", "decode1", "decode2", "decode3", "frontend"]:
            log_path = f"/tmp/dynamo-{worker_name}.log"
            log_files[worker_name] = open(log_path, "w")
            print(f"   {worker_name} -> {log_path}")

    # Set model path from arguments
    MODEL_PATH = Path(args.model_path)

    # Detect GPUs
    num_gpus = get_num_gpus()
    print(f"🔍 Detected {num_gpus} GPU(s)")
    print(f"📦 Model path: {MODEL_PATH}")

    if num_gpus == 0:
        print("❌ No GPUs detected, exiting...")
        return 1

    # Determine worker configuration based on GPU count.
    # Distribute 4 workers (1 prefill + 3 decode) across available GPUs as evenly as possible.
    num_decode_workers = len(TIER_BOUNDARIES)
    num_workers = 1 + num_decode_workers  # 1 prefill + N decode
    worker_names = ["prefill"] + [f"decode{i+1}" for i in range(num_decode_workers)]

    # Assign GPUs to workers (round-robin distribution)
    gpu_assignments = {}
    if num_gpus == 1:
        # All workers share GPU 0
        for worker_name in worker_names:
            gpu_assignments[worker_name] = "0"
        print(f"🚀 Using 1 GPU: all {num_workers} workers on GPU 0")
    else:
        # Distribute workers across available GPUs
        for i, worker_name in enumerate(worker_names):
            gpu_id = i % num_gpus
            gpu_assignments[worker_name] = str(gpu_id)
        print(
            f"🚀 Using {num_gpus} GPUs: distributing {num_workers} workers across GPUs"
        )
        for worker_name, gpu_id in gpu_assignments.items():
            print(f"   {worker_name} → GPU {gpu_id}")

    # Count workers per GPU and calculate mem-fraction for each worker
    from collections import Counter

    workers_per_gpu = Counter(gpu_assignments.values())
    worker_mem_fractions = {}
    base_fraction = 0.8  # Total fraction to use when multiple workers share a GPU

    for worker_name, gpu_id in gpu_assignments.items():
        num_workers_on_gpu = workers_per_gpu[gpu_id]
        worker_mem_fractions[worker_name] = base_fraction / num_workers_on_gpu

    print("📊 Memory fractions per worker:")
    for worker_name in worker_names:
        gpu_id = gpu_assignments[worker_name]
        mem_frac = worker_mem_fractions[worker_name]
        print(f"   {worker_name} (GPU {gpu_id}): {mem_frac:.3f}")

    # SGLang uses server_args.port as the base for several internal ports.
    # When DP attention is enabled, it derives ports like:
    #   - nccl_port = port + random(100, 1000)
    #   - dist_init = port + 233
    # So workers need at least 1100 port spacing to avoid nccl_port conflicts.
    base_sglang_port = 10000
    sglang_port_stride = 1500  # keep ranges apart to avoid nccl_port collisions
    worker_ports = {
        "prefill": base_sglang_port,
        "decode1": base_sglang_port + 1 * sglang_port_stride,
        "decode2": base_sglang_port + 2 * sglang_port_stride,
        "decode3": base_sglang_port + 3 * sglang_port_stride,
    }
    # Each worker needs a unique disaggregation bootstrap port
    base_bootstrap_port = 12345
    bootstrap_ports = {
        "prefill": base_bootstrap_port,
        "decode1": base_bootstrap_port + 1,
        "decode2": base_bootstrap_port + 2,
        "decode3": base_bootstrap_port + 3,
    }
    # Each worker needs an explicit NCCL port to avoid random collisions
    base_nccl_port = 29500
    nccl_ports = {
        "prefill": base_nccl_port,
        "decode1": base_nccl_port + 100,
        "decode2": base_nccl_port + 200,
        "decode3": base_nccl_port + 300,
    }

    # For simplicity with multiple GPUs, use single GPU per worker (no TP/DP)
    prefill_tp_args = []
    decode_dp_args = []

    try:
        # Start prefill worker
        prefill_env = {"DYN_SYSTEM_PORT": "8081"}
        prefill_env["CUDA_VISIBLE_DEVICES"] = gpu_assignments["prefill"]

        start_worker(
            "prefill",
            [
                sys.executable,
                "-m",
                "dynamo.sglang",
                *get_common_args(MODEL_PATH, worker_mem_fractions["prefill"]),
                *prefill_tp_args,
                "--load-balance-method",
                "round_robin",
                "--disaggregation-mode",
                "prefill",
                "--disaggregation-bootstrap-port",
                str(bootstrap_ports["prefill"]),
                "--disaggregation-transfer-backend",
                "nixl",
                "--kv-transfer-method",
                "triton",
                "--host",
                "0.0.0.0",
                "--port",
                str(worker_ports["prefill"]),
                "--nccl-port",
                str(nccl_ports["prefill"]),
            ],
            env=prefill_env,
        )

        # Start decode workers
        for i in range(num_decode_workers):
            worker_name = f"decode{i + 1}"
            seqlen_tier = TIER_BOUNDARIES[i]
            system_port = 8082 + i

            worker_env = {"DYN_SYSTEM_PORT": str(system_port)}
            worker_env["CUDA_VISIBLE_DEVICES"] = gpu_assignments[worker_name]

            start_worker(
                worker_name,
                [
                    sys.executable,
                    "-m",
                    "dynamo.sglang",
                    *get_common_args(MODEL_PATH, worker_mem_fractions[worker_name]),
                    *decode_dp_args,
                    "--prefill-round-robin-balance",
                    "--disaggregation-mode",
                    "decode",
                    "--disaggregation-bootstrap-port",
                    str(bootstrap_ports[worker_name]),
                    "--disaggregation-transfer-backend",
                    "nixl",
                    "--kv-transfer-method",
                    "triton",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(worker_ports[worker_name]),
                    "--nccl-port",
                    str(nccl_ports[worker_name]),
                    "--this-seqlen",
                    str(seqlen_tier),
                ],
                env=worker_env,
            )

        # Start frontend
        start_worker(
            "frontend",
            [
                sys.executable,
                "-m",
                "dynamo.frontend",
                "--http-port",
                "8080",
                "--namespace",
                "dynamo",
                "--enable-decode-disagg",
                "--enforce-disagg",
            ],
        )

        # Start process monitor thread
        monitor_thread = threading.Thread(target=monitor_processes, daemon=True)
        monitor_thread.start()

        # Wait for health
        if not wait_for_health(num_workers):
            print("Failed to start workers, exiting...")
            return 1

        # Run automated requests in claude mode, otherwise interactive
        if claude_mode:
            run_automated_requests()
        else:
            interactive_loop(MODEL_PATH)

    finally:
        stop_event.set()  # Signal monitor to stop
        cleanup()

        # Close log files in claude mode
        if claude_mode:
            for log_file in log_files.values():
                log_file.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
