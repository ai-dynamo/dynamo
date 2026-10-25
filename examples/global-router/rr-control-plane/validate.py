# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
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
"""Run inside a cluster with read-only pod/log access to validate discovery and round robin."""

import collections
import json
import os
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path

NAMESPACE = os.environ.get("POD_NAMESPACE", "global-router-rr")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-32B")
EXPECTED = int(os.environ.get("EXPECTED_POOLS", "4"))
COUNT = int(os.environ.get("REQUEST_COUNT", "40"))
if EXPECTED <= 0 or COUNT < EXPECTED:
    raise ValueError("request count must cover every expected pool")
TOKEN = Path("/var/run/secrets/kubernetes.io/serviceaccount/token").read_text().strip()
TLS = ssl.create_default_context(
    cafile="/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
)
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def get(url):
    with HTTP.open(url, timeout=30) as response:
        return json.load(response)


def kube(path):
    request = urllib.request.Request(
        "https://kubernetes.default.svc" + path,
        headers={"Authorization": "Bearer " + TOKEN},
    )
    with urllib.request.urlopen(request, context=TLS, timeout=30) as response:
        return response.read().decode()


def logs():
    pods = json.loads(
        kube(f"/api/v1/namespaces/{NAMESPACE}/pods?labelSelector=app%3Dglobal-router")
    )["items"]
    assert len(pods) == 1, "validation expects exactly one ingress replica"
    name = pods[0]["metadata"]["name"]
    text = kube(
        f"/api/v1/namespaces/{NAMESPACE}/pods/{name}/log?container=global-router"
    )
    counts = collections.Counter()
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        fields = event.get("fields", event)
        if fields.get("message") == "forwarding request":
            counts[fields["url"].split("/v1/")[0] + "/"] += 1
    return counts


def run():
    deadline = time.monotonic() + 180
    previous_inventory = None
    stable_checks = 0
    while True:
        if time.monotonic() > deadline:
            raise TimeoutError("discovery inventory did not stabilize")
        try:
            pools = get("http://global-discovery:8080/pools")
            routers = get("http://global-discovery:8080/routers")["routers"]
            assert pools["complete"] and len(pools["pools"]) == EXPECTED
            assert len(routers) == EXPECTED
            assert all(
                p["model"] == MODEL and len(p["frontends"]) == 1 for p in pools["pools"]
            )
            assert MODEL in [
                m["id"] for m in get("http://global-router:8000/v1/models")["data"]
            ]
            inventory = json.dumps(
                sorted(pools["pools"], key=lambda p: p["key"]["site_id"]),
                sort_keys=True,
            )
            stable_checks = stable_checks + 1 if inventory == previous_inventory else 1
            previous_inventory = inventory
            # Membership can reach GDS before ingress's next two-second poll.
            # Require a stable view across six seconds before measuring balance.
            if stable_checks >= 3:
                break
        except Exception:
            if time.monotonic() > deadline:
                raise
            stable_checks = 0
        time.sleep(3)
    endpoints = {p["frontends"][0]: p["key"]["site_id"] for p in pools["pools"]}
    for router in routers:
        assert MODEL in [
            m["id"] for m in get(router["url"].rstrip("/") + "/v1/models")["data"]
        ]
    before = logs()
    elapsed = []
    for i in range(COUNT):
        stream = i % 2 == 0
        chat = i % 4 < 2
        body = {"model": MODEL, "max_tokens": 4, "stream": stream, "temperature": 0}
        route = "chat/completions" if chat else "completions"
        body.update(
            {"messages": [{"role": "user", "content": "Say hello."}]}
            if chat
            else {"prompt": "Say hello."}
        )
        request = urllib.request.Request(
            "http://global-router:8000/v1/" + route,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        start = time.monotonic()
        with HTTP.open(request, timeout=45) as response:
            assert response.status == 200
            result = response.read().decode()
            if stream:
                assert "data: [DONE]" in result
                chunks = [
                    json.loads(line[6:])
                    for line in result.splitlines()
                    if line.startswith("data: {")
                ]
                assert chunks and not any("error" in c for c in chunks)
            else:
                decoded = json.loads(result)
                assert decoded["choices"] and "error" not in decoded
        elapsed.append(round((time.monotonic() - start) * 1000, 1))
    after = logs()
    routed = after - before
    assert sum(routed.values()) == COUNT, (routed, COUNT)
    assert set(routed) == set(endpoints), (routed, endpoints)
    counts = {endpoints[url]: count for url, count in routed.items()}
    assert max(counts.values()) - min(counts.values()) <= 1, counts
    return {
        "success": True,
        "routers": len(routers),
        "pools": len(pools["pools"]),
        "requests": COUNT,
        "counts_by_site": counts,
        "http_200": COUNT,
        "frontend_urls_by_site": {site: url for url, site in endpoints.items()},
        "streaming": (COUNT + 1) // 2,
        "elapsed_ms": elapsed,
    }


if __name__ == "__main__":
    try:
        result = run()
    except Exception as error:
        result = {"success": False, "error": repr(error)}
        Path("/dev/termination-log").write_text(json.dumps(result))
        print(json.dumps(result), flush=True)
        raise
    Path("/dev/termination-log").write_text(json.dumps(result))
    print(json.dumps(result), flush=True)
