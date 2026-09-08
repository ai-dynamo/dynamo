# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Black-box N-2 tests. Only the HTTP client runs from the current checkout."""

import argparse
import json
import math
import re
import subprocess
import time
import uuid
from pathlib import Path

import requests


def command(*args, timeout=600):
    return subprocess.check_output(
        args, text=True, stderr=subprocess.STDOUT, timeout=timeout
    )


def matrix(releases, line, frontend, worker):
    major, minor = map(int, line.split("."))
    if minor < 2:
        raise ValueError("Explicit same-major N-2 window requires minor >= 2")
    pairs = [("candidate", frontend, worker)]
    for age in (1, 2):
        previous = releases[f"{major}.{minor - age}"]
        pairs.extend(
            [
                (f"old-frontend-{age}", previous["frontend"], worker),
                (f"old-worker-{age}", frontend, previous["worker"]),
            ]
        )
    return pairs


def validate_embedding(body, count, dimensions):
    assert body["object"] == "list", body
    assert len(body["data"]) == count, body
    for index, item in enumerate(body["data"]):
        assert item["index"] == index, item
        assert item["object"] == "embedding", item
        vector = item["embedding"]
        # Do not decode strings here: float requests must return JSON arrays.
        assert isinstance(
            vector, list
        ), f"Expected float array, got {type(vector).__name__}"
        assert len(vector) == dimensions, len(vector)
        assert all(type(x) in (int, float) and math.isfinite(x) for x in vector)


def validate_chat(body, max_tokens, stop=None):
    assert "error" not in body, body
    assert len(body["choices"]) == 1, body
    choice = body["choices"][0]
    assert choice["index"] == 0, choice
    assert choice["finish_reason"] in ("stop", "length"), choice
    message = choice["message"]
    assert message["role"] == "assistant", message
    assert isinstance(message["content"], str), body
    if max_tokens > 1 and stop is None:
        assert message["content"].strip(), body
    assert 0 <= body["usage"]["completion_tokens"] <= max_tokens, body
    if stop is not None:
        assert message["content"] == "", body  # Stop is a prefix of the baseline.
        assert choice["finish_reason"] == "stop", body


def validate_stream(lines):
    content, finished, done = [], False, False
    for line in lines:
        if not line or line.startswith(":"):
            continue
        if line.startswith("event:"):
            assert line.strip() != "event: error", line
            continue
        assert line.startswith("data:"), f"Unexpected SSE line: {line}"
        assert not done, "Data after [DONE]"
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            continue
        chunk = json.loads(data)
        assert "error" not in chunk, chunk
        for choice in chunk["choices"]:
            assert choice["index"] == 0, choice
            text = choice.get("delta", {}).get("content")
            if text:
                assert isinstance(text, str)
                assert not finished, "Content after finish_reason"
                content.append(text)
            if choice.get("finish_reason") is not None:
                assert choice["finish_reason"] in ("stop", "length"), choice
                finished = True
    assert done and finished, "Incomplete SSE response"
    assert "".join(content).strip(), "Empty streamed content"


def probe(base, scenario, model, directory):
    cases = []
    if scenario == "embedding":
        for name, inputs, explicit in [
            ("default", "hello", False),
            ("float", "hello", True),
            ("batch", ["hello", "world"], True),
        ]:
            body = {"model": model["id"], "input": inputs}
            if explicit:
                body["encoding_format"] = "float"
            cases.append((name, "/v1/embeddings", body))
    else:
        for name, streaming, cap in [
            ("unary", False, 32),
            ("stream", True, 32),
            ("limited", False, 1),
            ("repeat", False, 32),
        ]:
            body = {
                "model": model["id"],
                "messages": [
                    {"role": "user", "content": "Reply with one short greeting."}
                ],
                "temperature": 0,
                "max_tokens": cap,
                "stream": streaming,
            }
            cases.append((name, "/v1/chat/completions", body))
    results = []
    for name, endpoint, body in cases:
        record = {"name": name, "request": body, "status": "failed"}
        try:
            with requests.post(
                base + endpoint,
                json=body,
                timeout=(10, 120),
                stream=body.get("stream", False),
            ) as response:
                record["http_status"] = response.status_code
                if body.get("stream"):
                    lines = []

                    def capture():
                        for line in response.iter_lines():
                            text = line.decode("utf-8")
                            lines.append(text)
                            yield text

                    try:
                        response.raise_for_status()
                        validate_stream(capture())
                    finally:
                        record["response"] = lines
                else:
                    record["response"] = response.text
                    response.raise_for_status()
                    value = response.json()
                    if scenario == "embedding":
                        count = (
                            len(body["input"]) if isinstance(body["input"], list) else 1
                        )
                        validate_embedding(value, count, model["dimensions"])
                    else:
                        validate_chat(value, body["max_tokens"], body.get("stop"))
                        if name == "unary":
                            # Derive a stop string from this deterministic response.
                            # No assumption about a small model following an instruction.
                            content = value["choices"][0]["message"]["content"]
                            stop = content[: min(len(content), 4)]
                            cases.append(("stop", endpoint, {**body, "stop": stop}))
            record["status"] = "passed"
        except (
            AssertionError,
            KeyError,
            TypeError,
            ValueError,
            requests.RequestException,
        ) as error:
            record["error"] = str(error)
        results.append(record)
        (directory / f"{name}.json").write_text(json.dumps(record, indent=2))
    return results


def wait_ready(base, model, alive, timeout=600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        alive()
        try:
            response = requests.get(base + "/v1/models", timeout=5)
            if response.ok and any(
                item["id"] == model for item in response.json()["data"]
            ):
                return
        except (requests.RequestException, ValueError, KeyError):
            pass
        time.sleep(2)
    raise TimeoutError(f"Model {model} was not discovered within {timeout}s")


class DockerStack:
    """Own every container/network, including resources from partial startup."""

    def __init__(self, directory):
        self.directory = directory
        self.network = "n2-" + uuid.uuid4().hex[:12]
        self.containers = []

    def __enter__(self):
        command("docker", "network", "create", self.network)
        return self

    def start(
        self, role, image, args, gpu=False, env=None, model_path=None, publish=False
    ):
        name = f"{self.network}-{role}"
        self.containers.append(name)
        options = [
            "docker",
            "create",
            "--name",
            name,
            "--network",
            self.network,
            "--network-alias",
            role,
            "--shm-size",
            "2g",
            "--workdir",
            "/tmp",
        ]
        if gpu:
            options += ["--gpus", "device=0"]
        if publish:
            # Container-local port; Docker allocates an isolated host port.
            options += ["-p", "127.0.0.1::8000"]
        for key, value in (env or {}).items():
            options += ["-e", f"{key}={value}"]
        command(*options, "--entrypoint", args[0], image, *args[1:])
        if model_path:
            # Works with DinD too: do not assume daemon/runner share bind paths.
            command("docker", "cp", str(model_path), f"{name}:/model")
        command("docker", "start", name)
        return name

    def alive(self):
        for name in self.containers:
            state = json.loads(command("docker", "inspect", name))[0]["State"]
            if not state["Running"]:
                raise RuntimeError(f"{name} exited: {state}")

    def __exit__(self, *exc):
        failures = []
        for name in reversed(self.containers):
            try:
                logs = command("docker", "logs", name, timeout=30)
                (self.directory / f"{name}.log").write_text(logs)
                metadata = command(
                    "docker", "inspect", "--format", "{{json .State}}", name, timeout=30
                )
                (self.directory / f"{name}-state.json").write_text(metadata)
            except (subprocess.SubprocessError, OSError) as error:
                failures.append(str(error))
            finally:
                try:
                    command("docker", "rm", "-f", name, timeout=30)
                except subprocess.SubprocessError as error:
                    failures.append(str(error))
        try:
            command("docker", "network", "rm", self.network, timeout=30)
        except subprocess.SubprocessError as error:
            failures.append(str(error))
        if failures:
            (self.directory / "cleanup-errors.json").write_text(json.dumps(failures))
            if exc[0] is None:
                raise RuntimeError(
                    "Container collection/cleanup failed; see cleanup-errors.json"
                )


def resolve_image(reference):
    command("docker", "pull", reference)
    image = json.loads(command("docker", "image", "inspect", reference))[0]
    return {
        "reference": reference,
        "id": image["Id"],
        "digests": image.get("RepoDigests", []),
    }


def execute(frontend, worker, infra, scenario, model, directory):
    with DockerStack(directory) as stack:
        stack.start(
            "etcd",
            infra["etcd"]["id"],
            [
                "etcd",
                "--advertise-client-urls",
                "http://etcd:2379",
                "--listen-client-urls",
                "http://0.0.0.0:2379",
            ],
        )
        stack.start("nats", infra["nats"]["id"], ["nats-server", "-js"])
        deadline = time.monotonic() + 60
        while True:
            stack.alive()
            try:
                command(
                    "docker",
                    "exec",
                    f"{stack.network}-etcd",
                    "etcdctl",
                    "endpoint",
                    "health",
                    timeout=10,
                )
                break
            except subprocess.SubprocessError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("etcd did not become healthy")
                time.sleep(1)
        env = {
            "ETCD_ENDPOINTS": "http://etcd:2379",
            "NATS_SERVER": "nats://nats:4222",
            "DYN_NAMESPACE": stack.network,
            "DYN_DISCOVERY_BACKEND": "etcd",
            "DYN_REQUEST_PLANE": "tcp",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DYN_SYSTEM_PORT": "8081",
        }
        path = model["path"]
        # Start from /tmp: never import a checkout mounted by CI.
        worker_args = [
            "python3",
            "-m",
            "dynamo.sglang",
            "--model-path",
            "/model",
            "--served-model-name",
            model["id"],
            "--tp",
            "1",
            "--mem-fraction-static",
            "0.65",
            "--enable-metrics",
        ]
        if scenario == "embedding":
            worker_args += [
                "--embedding-worker",
                "--use-sglang-tokenizer",
                "--page-size",
                "16",
            ]
        stack.start(
            "worker", worker["id"], worker_args, gpu=True, env=env, model_path=path
        )
        name = stack.start(
            "frontend",
            frontend["id"],
            ["python3", "-m", "dynamo.frontend", "--http-port", "8000"],
            env={**env, "DYN_SYSTEM_PORT": "8082"},
            model_path=path,
            publish=True,
        )
        address = command("docker", "port", name, "8000/tcp").strip()
        base = "http://" + address
        for role in ("worker", "frontend"):
            version = command(
                "docker",
                "exec",
                f"{stack.network}-{role}",
                "python3",
                "-c",
                "import importlib.metadata as m,json; print(json.dumps({"
                'd.metadata["Name"]:d.version for d in m.distributions() '
                'if d.metadata["Name"] in ("ai-dynamo", "ai-dynamo-runtime", "sglang")}))',
            )
            (directory / f"{role}-versions.json").write_text(version)
        wait_ready(base, model["id"], stack.alive)
        return probe(base, scenario, model, directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("releases.json")
    )
    parser.add_argument(
        "--release-line", default="", help="Default: checkout Cargo.toml version"
    )
    parser.add_argument("--frontend-image", required=True)
    parser.add_argument("--worker-image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--plan", action="store_true", help="Validate/print matrix without Docker"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not args.release_line:
        cargo = Path(__file__).resolve().parents[2] / "Cargo.toml"
        args.release_line = re.search(
            r'^version = "(\d+\.\d+)\.', cargo.read_text(), re.M
        )[1]
    pairs = matrix(
        config["releases"], args.release_line, args.frontend_image, args.worker_image
    )
    if args.plan:
        print(json.dumps(pairs, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "release_line": args.release_line,
        "pairs": pairs,
        "status": "failed",
        "runs": [],
    }
    try:
        command("docker", "info", timeout=30)
        refs = {ref for _, frontend, worker in pairs for ref in (frontend, worker)}
        refs.update(config["infrastructure"].values())
        images = {ref: resolve_image(ref) for ref in sorted(refs)}
        report["images"] = images
        # Freeze model revisions once, then copy the same snapshot in every version.
        from huggingface_hub import HfApi, snapshot_download

        models = {}
        for scenario, spec in config["models"].items():
            revision = HfApi().model_info(spec["id"], revision=spec["revision"]).sha
            path = args.output.resolve() / "models" / scenario
            snapshot_download(spec["id"], revision=revision, local_dir=path)
            models[scenario] = {**spec, "revision": revision, "path": str(path)}
        report["models"] = models
        infra = {name: images[ref] for name, ref in config["infrastructure"].items()}
        for name, frontend, worker in pairs:
            for scenario, model in models.items():
                directory = args.output / name / scenario
                directory.mkdir(parents=True)
                run = {"pair": name, "scenario": scenario, "status": "failed"}
                report["runs"].append(run)
                try:
                    run["cases"] = execute(
                        images[frontend],
                        images[worker],
                        infra,
                        scenario,
                        model,
                        directory,
                    )
                    if all(case["status"] == "passed" for case in run["cases"]):
                        run["status"] = "passed"
                except (
                    subprocess.SubprocessError,
                    OSError,
                    RuntimeError,
                    ValueError,
                ) as error:
                    run["error"] = str(error)
                finally:
                    (args.output / "report.json").write_text(
                        json.dumps(report, indent=2)
                    )
        if all(run["status"] == "passed" for run in report["runs"]):
            report["status"] = "passed"
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit("Cross-version compatibility failed; see report.json")


if __name__ == "__main__":
    main()
