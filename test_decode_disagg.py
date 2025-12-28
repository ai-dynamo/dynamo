"""Test script for decode disaggregation with interactive UI."""

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

NUM_WORKERS = 3  # 1 prefill + 2 decode
HEALTH_URL = "http://localhost:8080/health"
CHAT_URL = "http://localhost:8080/v1/chat/completions"
MODEL_PATH = Path.home() / "proj/models/smol2-135m"

# Event to signal all processes should stop
stop_event = threading.Event()

# Track processes with names for monitoring
processes: dict[str, subprocess.Popen] = {}

# ANSI colors for log prefixes
COLORS = {
    "prefill": "\033[32m",  # green
    "decode1": "\033[36m",  # cyan
    "decode2": "\033[35m",  # magenta
    "frontend": "\033[33m",  # yellow
    "reset": "\033[0m",
}

COMMON_ARGS = [
    "--model-path",
    str(MODEL_PATH),
    "--context-length",
    "512",
    "--mem-fraction-static",
    "0.33",
    "--page-size",
    "16",
    "--disable-cuda-graph",
]


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


def wait_for_health(timeout: float = 120.0):
    """Poll /health until we have the expected number of instances."""
    print(f"\n⏳ Waiting for {NUM_WORKERS} workers to be ready...")
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

            if count >= NUM_WORKERS:
                print(f"✅ All {NUM_WORKERS} workers are ready!")
                return True
            else:
                print(f"   ... {count}/{NUM_WORKERS} workers ready", end="\r")
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


def run_request(max_tokens: int):
    """Run a chat completion request with streaming."""
    prompt = random_prompt()
    print(f"\n📝 Prompt: {prompt}")
    print(f"🎯 Max tokens: {max_tokens}")
    print("-" * 50)

    payload = {
        "model": str(MODEL_PATH),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
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
        print(final_text)
        print("-" * 50)
    except Exception as e:
        print(f"\n❌ Request failed: {e}")


def interactive_loop():
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
                run_request(osl)
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
    try:
        # Start prefill worker
        start_worker(
            "prefill",
            [
                sys.executable,
                "-m",
                "dynamo.sglang",
                *COMMON_ARGS,
                "--disaggregation-mode",
                "prefill",
                "--disaggregation-bootstrap-port",
                "12345",
                "--disaggregation-transfer-backend",
                "nixl",
                "--host",
                "0.0.0.0",
            ],
            env={"DYN_SYSTEM_PORT": "8081"},
        )

        # Start decode worker 1 (handles seqlen <= 128)
        start_worker(
            "decode1",
            [
                sys.executable,
                "-m",
                "dynamo.sglang",
                *COMMON_ARGS,
                "--disaggregation-mode",
                "decode",
                "--disaggregation-bootstrap-port",
                "12346",
                "--disaggregation-transfer-backend",
                "nixl",
                "--host",
                "0.0.0.0",
                "--this-seqlen",
                "128",
            ],
            env={"DYN_SYSTEM_PORT": "8082"},
        )

        # Start decode worker 2 (handles seqlen <= 1024)
        start_worker(
            "decode2",
            [
                sys.executable,
                "-m",
                "dynamo.sglang",
                *COMMON_ARGS,
                "--disaggregation-mode",
                "decode",
                "--disaggregation-bootstrap-port",
                "12347",
                "--disaggregation-transfer-backend",
                "nixl",
                "--host",
                "0.0.0.0",
                "--this-seqlen",
                "512",
            ],
            env={"DYN_SYSTEM_PORT": "8083"},
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
        if not wait_for_health():
            print("Failed to start workers, exiting...")
            return 1

        # Interactive loop
        interactive_loop()

    finally:
        stop_event.set()  # Signal monitor to stop
        cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
