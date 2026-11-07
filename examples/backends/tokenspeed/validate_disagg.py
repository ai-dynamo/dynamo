# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify live LongCat P/D generation and cache-aware selection across two prefill replicas."""

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path

import httpx
from transformers import AutoTokenizer

from dynamo.llm import KvRouter, KvRouterConfig
from dynamo.runtime import DistributedRuntime


async def wait_for_frontend(http, url, model, timeout=120):
    """Wait for model registration, including while the frontend port is closed."""
    deadline = time.monotonic() + timeout
    last_observation = "no response"
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                f"Frontend did not expose {model!r}; inspect its model registration "
                f"logs. Last /v1/models result: {last_observation}"
            )
        try:
            response = await http.get(
                url.rstrip("/") + "/v1/models", timeout=min(5.0, remaining)
            )
            last_observation = f"{response.status_code}: {response.text[:1000]}"
            if response.status_code == 200 and any(
                row.get("id") == model for row in response.json().get("data", [])
            ):
                return
        except httpx.TransportError as error:
            last_observation = f"{type(error).__name__}: {error}"
        await asyncio.sleep(min(0.5, max(0, deadline - time.monotonic())))


async def run(args):
    """Check live generation and prefix routing, persisting the validation report."""
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    runtime = DistributedRuntime(
        asyncio.get_running_loop(), "etcd", "tcp", event_plane="nats"
    )
    observations = []
    report = {"model": args.model, "requests": observations, "passed": False}
    try:
        endpoint = runtime.endpoint(f"{args.namespace}.prefill.generate")
        client = await endpoint.client()
        deadline = time.monotonic() + 60
        while len(client.instance_ids()) < 2 and time.monotonic() < deadline:
            await asyncio.sleep(0.25)
        prefill_ids = sorted(client.instance_ids())
        if len(prefill_ids) != 2:
            raise RuntimeError(f"Expected two prefill workers, found {prefill_ids}")
        report["prefill_workers"] = prefill_ids
        router = KvRouter(endpoint, 64, KvRouterConfig(use_kv_events=True))
        nonce = uuid.uuid4().hex
        prompts = []
        for topic in ("ORANGE", "PURPLE", "SILVER"):
            content = (
                f"Document {topic} {nonce}.\n"
                + (f"The identifying word for this document is {topic}. " * 80)
                + "\nAnswer with the identifying word only."
            )
            prompts.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=False,
                    enable_thinking=False,
                )
            )

        async def scores(tokens):
            """Read each prefill worker's cached-prefix overlap."""
            return await router.get_overlap_scores(tokens)

        async def wait_for_cache(tokens, owner):
            """Wait until the router observes the owner's complete cached prefix."""
            deadline = time.monotonic() + 30
            expected = len(tokens) // 64
            while True:
                result = await scores(tokens)
                hits = {
                    row["worker_id"]: row["device_blocks"] for row in result["workers"]
                }
                if hits.get(owner, 0) >= expected:
                    return result
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Expected {expected} cached blocks on {owner}: {result}"
                    )
                await asyncio.sleep(0.25)

        async with httpx.AsyncClient(timeout=300) as http:
            await wait_for_frontend(http, args.url, args.model)

            async def request(label, tokens, expected_word, forced_prefill=None):
                """Generate one response and validate its completion and worker metadata."""
                headers = {"x-request-id": f"longcat-{label}-{uuid.uuid4().hex}"}
                if forced_prefill is not None:
                    headers["x-dynamo-prefill-instance-id"] = str(forced_prefill)
                payload = {
                    "model": args.model,
                    "prompt": tokens,
                    "max_tokens": 32,
                    "temperature": 0,
                    "stream": True,
                    "nvext": {"extra_fields": ["worker_id"]},
                }
                text, workers, chunks, finished = "", {}, [], False
                started = time.monotonic()
                async with http.stream(
                    "POST",
                    args.url.rstrip("/") + "/v1/completions",
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status_code != 200:
                        body = b""
                        async for chunk in response.aiter_bytes(chunk_size=1000):
                            body = chunk
                            break
                        raise RuntimeError(
                            f"{label}: HTTP {response.status_code}: "
                            f"{body.decode(errors='replace')}"
                        )
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            finished = True
                            continue
                        chunk = json.loads(data)
                        chunks.append(chunk)
                        if chunk.get("error"):
                            raise RuntimeError(f"{label}: {chunk['error']}")
                        workers.update(chunk.get("nvext", {}).get("worker_id", {}))
                        for choice in chunk.get("choices", []):
                            text += choice.get("text", "")
                record = {
                    "label": label,
                    "elapsed_s": time.monotonic() - started,
                    "worker_ids": workers,
                    "text": text,
                    "chunks": chunks,
                    "forced_prefill": forced_prefill,
                    "expected_word": expected_word,
                    "prompt_tokens": len(tokens),
                }
                observations.append(record)
                if not finished or expected_word not in text.upper():
                    raise RuntimeError(f"Incomplete or incorrect generation: {record}")
                if workers.get("prefill_worker_id") not in prefill_ids:
                    raise RuntimeError(f"Unexpected prefill worker: {record}")
                if not isinstance(workers.get("decode_worker_id"), int):
                    raise RuntimeError(f"Missing decode worker: {record}")
                if (
                    forced_prefill is not None
                    and workers["prefill_worker_id"] != forced_prefill
                ):
                    raise RuntimeError(
                        f"Forced prefill worker was not selected: {record}"
                    )
                print(
                    json.dumps({k: v for k, v in record.items() if k != "chunks"}),
                    flush=True,
                )
                return record

            for i, owner in enumerate(prefill_ids):
                await request(
                    f"warm-{i}",
                    prompts[i],
                    ("ORANGE", "PURPLE")[i],
                    forced_prefill=owner,
                )
                report[f"warm_{i}_overlap"] = await wait_for_cache(prompts[i], owner)
            report["cold_overlap"] = await scores(prompts[2])
            if any(
                row["device_blocks"] != 0 for row in report["cold_overlap"]["workers"]
            ):
                raise RuntimeError(
                    f"Cold prefix was already cached: {report['cold_overlap']}"
                )
            # Consecutive requests for the same owner distinguish cache-aware
            # selection from round-robin, which can pass alternating probes.
            for i, owner in enumerate(prefill_ids):
                for repeat in range(2):
                    record = await request(
                        f"reuse-{i}-{repeat}", prompts[i], ("ORANGE", "PURPLE")[i]
                    )
                    if record["worker_ids"]["prefill_worker_id"] != owner:
                        raise RuntimeError(
                            f"Cached-prefix owner was not selected: {record}"
                        )
            await request("cold-control", prompts[2], "SILVER")
        report["passed"] = True
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        runtime.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="longcat-flash")
    parser.add_argument("--namespace", default="tokenspeed")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True, type=Path)
    asyncio.run(run(parser.parse_args()))
