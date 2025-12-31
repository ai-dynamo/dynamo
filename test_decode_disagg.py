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

TIER_BOUNDARIES = [128, 256, 512]  # Token boundaries for decode tiers
HEALTH_URL = "http://localhost:8080/health"
CHAT_URL = "http://localhost:8080/v1/chat/completions"

# Event to signal all processes should stop
stop_event = threading.Event()

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


def get_common_args(num_gpus: int, model_path: Path) -> list[str]:
    """Get common arguments, adjusting mem-fraction-static based on GPU count."""
    mem_fraction = "0.8" if num_gpus > 1 else "0.15"
    return [
        "--model-path",
        str(model_path),
        "--context-length",
        "512",
        "--mem-fraction-static",
        mem_fraction,
        "--page-size",
        "16",
        "--disable-cuda-graph",
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
    """Stream stdout/stderr from a process with a colored prefix."""
    color = COLORS.get(name, "")
    reset = COLORS["reset"]
    prefix = f"{color}[{name}]{reset} "

    pipe = proc.stdout if stream == "stdout" else proc.stderr
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


def wait_for_health(num_workers: int, timeout: float = 120.0):
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
            count = len(instances)

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
    print(f"\n📝 Prompt: {prompt}")
    print(f"🎯 Max tokens: {max_tokens}")
    print("-" * 50)

    payload = {
        "model": str(model_path),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "ignore_eos": True,  # Always generate exactly max_tokens for debugging
    }

    try:
        full_output = []
        print("⏳ Generating... (output will be shown at the end)")
        with httpx.stream("POST", CHAT_URL, json=payload, timeout=120.0) as resp:
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
        default=str(Path.home() / "proj/models/smol2-135m"),
        help="Path to the model directory (default: ~/proj/models/smol2-135m)",
    )
    args = parser.parse_args()

    # Set model path from arguments
    MODEL_PATH = Path(args.model_path)

    # Detect GPUs
    num_gpus = get_num_gpus()
    print(f"🔍 Detected {num_gpus} GPU(s)")
    print(f"📦 Model path: {MODEL_PATH}")

    if num_gpus == 0:
        print("❌ No GPUs detected, exiting...")
        return 1

    # Get common args based on GPU count
    common_args = get_common_args(num_gpus, MODEL_PATH)
    mem_fraction = "0.8" if num_gpus > 1 else "0.15"
    print(f"📊 Using mem-fraction-static: {mem_fraction}")

    # Determine worker configuration based on GPU count
    # GPU 0: prefill worker
    # GPU 1+: decode workers with different seqlen tiers
    decode_tiers = [128, 256, 512]  # seqlen tiers for decode workers
    num_decode_workers = (
        min(num_gpus - 1, len(decode_tiers)) if num_gpus > 1 else len(decode_tiers)
    )
    num_workers = 1 + num_decode_workers  # 1 prefill + N decode

    print(f"🚀 Starting {num_workers} workers: 1 prefill + {num_decode_workers} decode")

    try:
        # Start prefill worker on GPU 0 (or all GPUs if only 1)
        prefill_env = {"DYN_SYSTEM_PORT": "8081"}
        if num_gpus > 1:
            prefill_env["CUDA_VISIBLE_DEVICES"] = "0"

        start_worker(
            "prefill",
            [
                sys.executable,
                "-m",
                "dynamo.sglang",
                *common_args,
                "--disaggregation-mode",
                "prefill",
                "--disaggregation-bootstrap-port",
                "12345",
                "--disaggregation-transfer-backend",
                "nixl",
                "--host",
                "0.0.0.0",
            ],
            env=prefill_env,
        )

        # Start decode workers
        for i in range(num_decode_workers):
            worker_name = f"decode{i + 1}"
            gpu_id = i + 1 if num_gpus > 1 else 0  # Use separate GPU if available
            seqlen_tier = decode_tiers[i]
            bootstrap_port = 12346 + i
            system_port = 8082 + i

            worker_env = {"DYN_SYSTEM_PORT": str(system_port)}
            if num_gpus > 1:
                worker_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            start_worker(
                worker_name,
                [
                    sys.executable,
                    "-m",
                    "dynamo.sglang",
                    *common_args,
                    "--disaggregation-mode",
                    "decode",
                    "--disaggregation-bootstrap-port",
                    str(bootstrap_port),
                    "--disaggregation-transfer-backend",
                    "nixl",
                    "--host",
                    "0.0.0.0",
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

        # Interactive loop
        interactive_loop(MODEL_PATH)

    finally:
        stop_event.set()  # Signal monitor to stop
        cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
