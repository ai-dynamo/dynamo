"""Production wiring primitives; every external operation is injected."""

import calendar
import datetime as dt
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import stat
import subprocess
import sys
import time


try:
    import v2_harness as _harness
except ModuleNotFoundError:
    _spec = importlib.util.spec_from_file_location(
        "v2_production_harness", pathlib.Path(__file__).with_name("v2_harness.py")
    )
    _harness = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _harness
    _spec.loader.exec_module(_harness)


RESTORE_CREATION_SKEW_S = 5.0
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class _RestorePending(ValueError):
    pass


class SubprocessTransport:
    def __init__(self, timeout_s):
        self.timeout_s = timeout_s

    def __call__(self, argv, timeout_s=None):
        return subprocess.run(
            list(argv), text=True, capture_output=True,
            timeout=self.timeout_s if timeout_s is None else timeout_s,
            shell=False, check=False,
        )


def _epoch(value):
    match = re.fullmatch(
        r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})",
        value if isinstance(value, str) else "",
    )
    if match is None:
        raise ValueError("invalid lifecycle timestamp")
    try:
        year, month, day, hour, minute, second = map(int, match.groups()[:6])
        whole = calendar.timegm(dt.datetime(year, month, day, hour, minute, second).timetuple())
        zone = match.group(8)
        if zone != "Z":
            direction = 1 if zone[0] == "+" else -1
            offset_hour, offset_minute = map(int, zone[1:].split(":"))
            if offset_hour > 23 or offset_minute > 59:
                raise ValueError("invalid lifecycle timestamp")
            whole -= direction * (offset_hour * 3600 + offset_minute * 60)
        fraction = int((match.group(7) or "").ljust(9, "0"))
        return whole + fraction / 1_000_000_000
    except ValueError as exc:
        raise ValueError("invalid lifecycle timestamp") from exc


def _json(result):
    if getattr(result, "returncode", 0) != 0:
        raise ValueError(getattr(result, "stderr", "transport failure") or "transport failure")
    try:
        return json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid JSON evidence") from exc


def _condition(pod, name):
    for item in pod.get("status", {}).get("conditions", []):
        if item.get("type") == name and item.get("status") == "True":
            return _epoch(item.get("lastTransitionTime"))
    raise ValueError("missing " + name + " condition")


def _seconds(value):
    if not isinstance(value, str) or not value:
        raise ValueError("invalid duration")
    scales = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "ns": 1e-9}
    rank = {"h": 6, "m": 5, "s": 4, "ms": 3, "us": 2, "µs": 2, "ns": 1}
    token = re.compile(r"([0-9]+(?:\.[0-9]+)?)(ns|µs|us|ms|h|m|s)")
    position = total = 0
    previous = 7
    terminal = None
    while position < len(value):
        match = token.match(value, position)
        if match is None:
            raise ValueError("invalid duration")
        number, unit = match.groups()
        if rank[unit] >= previous:
            raise ValueError("invalid duration")
        total += float(number) * scales[unit]
        previous = rank[unit]
        terminal = unit
        position = match.end()
    if total <= 0 or terminal in {"h", "m"}:
        raise ValueError("invalid duration")
    return total


def _restore_log(logs, target, container_id, created, expected_checkpoint_id=None):
    """Return the sole current-container restore start/summary pair."""
    pairs = {}
    stale_target = False
    for line in logs.splitlines():
        marker = None
        if "=== Starting external restore ===" in line:
            marker = "=== Starting external restore ==="
            kind = "start"
        elif "Restore timing summary" in line:
            marker = "Restore timing summary"
            kind = "summary"
        if marker is None:
            continue
        prefix, payload = line.split(marker, 1)
        timestamp = prefix.strip().split(None, 1)[0] if prefix.strip() else ""
        payload = payload[payload.find("{"):]
        try:
            when = _epoch(timestamp)
            item = json.loads(payload.strip(), object_pairs_hook=_restore_json_object)
        except (IndexError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("invalid snapshot-agent restore record") from exc
        if not isinstance(item, dict):
            continue
        short_target = target.rsplit("/", 1)[-1]
        namespace = target.split("/", 1)[0]
        duplicates = getattr(item, "duplicates", {})
        for key, values in duplicates.items():
            if key != "pod" and any(value != values[0] for value in values[1:]):
                raise ValueError("conflicting duplicate JSON field")
        pod_values = duplicates.get("pod", [item.get("pod")])
        if not all(isinstance(value, str) for value in pod_values):
            raise ValueError("invalid target pod identity")
        valid_pods = {target, short_target}
        if any(value not in valid_pods for value in pod_values):
            if any(value in valid_pods for value in pod_values):
                raise ValueError("conflicting target pod identity")
            continue
        if short_target in pod_values and item.get("namespace") != namespace:
            raise ValueError("target restore namespace identity mismatch")
        exact_target = True
        if not exact_target:
            continue
        checkpoint_id = item.get("checkpoint_id")
        if expected_checkpoint_id is not None and checkpoint_id != expected_checkpoint_id:
            raise ValueError("target restore checkpoint identity mismatch")
        if when < created - RESTORE_CREATION_SKEW_S:
            stale_target = True
            continue
        observed_id = item.get("container_id")
        if observed_id is not None and (not isinstance(observed_id, str) or not observed_id):
            raise ValueError("invalid target container identity")
        if container_id is not None:
            if observed_id not in {container_id, "containerd://" + container_id}:
                raise ValueError("target restore container identity mismatch")
        elif observed_id is not None:
            # A live Pod without a server container ID cannot safely bind a
            # container-scoped agent record.
            continue
        pair = pairs.setdefault(observed_id, {"starts": [], "summaries": []})
        if kind == "start":
            pair["starts"].append(when)
        else:
            try:
                candidate = item["restore"]["phases"]
            except (KeyError, TypeError) as exc:
                raise ValueError("invalid target restore timing") from exc
            if not isinstance(candidate, dict):
                raise ValueError("invalid target restore timing")
            pair["summaries"].append((when, candidate))
    candidates = []
    for pair in pairs.values():
        if len(set(pair["starts"])) > 1 or len({(when, json.dumps(phases, sort_keys=True)) for when, phases in pair["summaries"]}) > 1:
            raise ValueError("ambiguous target restore pair")
        if len(pair["starts"]) != 1 or len(pair["summaries"]) != 1:
            if not pair["starts"] or not pair["summaries"]:
                continue
        started = pair["starts"][0]
        summary_time, phases = pair["summaries"][0]
        if summary_time < started:
            raise ValueError("restore summary precedes start")
        candidates.append((started, summary_time, phases))
    if len(candidates) != 1:
        if stale_target:
            raise ValueError("target restore record is outside creation skew")
        raise _RestorePending("target restore pair is incomplete")
    return candidates[0]


def _identical_json_object(pairs):
    """Permit redundant JSON fields only when their decoded values are identical."""
    value = {}
    for key, item in pairs:
        if key in value and value[key] != item:
            raise ValueError("conflicting duplicate JSON field")
        value[key] = item
    return value


class _RestoreJSON(dict):
    """Keep duplicate top-level restore fields for identity normalization."""

    def __init__(self, pairs):
        super().__init__()
        self.duplicates = {}
        for key, item in pairs:
            if key in self:
                self.duplicates.setdefault(key, [self[key]]).append(item)
            self[key] = item


def _restore_json_object(pairs):
    return _RestoreJSON(pairs)


def _restore_events(events, namespace, pod_name, pod_uid, checkpoint_id):
    if not isinstance(events, dict) or not isinstance(events.get("items"), list):
        raise ValueError("invalid restore events")
    matched = []
    for item in events["items"]:
        involved = item.get("involvedObject") if isinstance(item, dict) else None
        if not isinstance(involved, dict):
            raise ValueError("invalid restore event")
        if (involved.get("uid"), involved.get("name"), involved.get("namespace")) != (pod_uid, pod_name, namespace):
            continue
        reason, message = item.get("reason"), item.get("message")
        if reason not in {"RestoreRequested", "RestoreSucceeded"}:
            continue
        expected = (
            "Restore requested from checkpoint " + checkpoint_id + " for container server"
            if reason == "RestoreRequested"
            else "Restore completed from checkpoint " + checkpoint_id
        )
        if message != expected:
            raise ValueError("restore event message mismatch")
        matched.append(reason)
    if matched.count("RestoreRequested") != 1 or matched.count("RestoreSucceeded") != 1 or len(matched) != 2:
        raise ValueError("missing or ambiguous restore events")
    return True


def _server_container_id(pod):
    statuses = pod.get("status", {}).get("containerStatuses", [])
    if not isinstance(statuses, list):
        raise ValueError("invalid container status")
    for status in statuses:
        if not isinstance(status, dict) or status.get("name") != "server":
            continue
        value = status.get("containerID")
        match = re.fullmatch(r"containerd://(.+)", value if isinstance(value, str) else "")
        if match is None:
            raise ValueError("server container lacks a containerd identity")
        return match.group(1)
    return None


def _probe_observation(value):
    fields = {
        "response", "http_200_epoch_s", "first_token_epoch_s", "tokens_per_second",
        "running", "waiting",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("invalid loopback observation")
    response = value["response"]
    if not isinstance(response, str) or not re.match(r"\s*2\b", response):
        raise ValueError("loopback response failed validity contract")
    for field in ("http_200_epoch_s", "first_token_epoch_s", "tokens_per_second"):
        if not isinstance(value[field], (int, float)) or isinstance(value[field], bool) or value[field] <= 0:
            raise ValueError("invalid loopback timing")
    for field in ("running", "waiting"):
        if not isinstance(value[field], int) or isinstance(value[field], bool) or value[field] < 0:
            raise ValueError("invalid loopback queue depth")
    if value["first_token_epoch_s"] < value["http_200_epoch_s"]:
        raise ValueError("out-of-order loopback timing")
    return value


class ProductionClusterPreflight:
    """Fail closed unless the reserved four-GPU host is exactly as pinned."""

    def __init__(self, namespace, node, reserves, agent, transport, *, timeout_s):
        if not isinstance(namespace, str) or not namespace or not isinstance(node, str) or not node:
            raise ValueError("invalid preflight target")
        if not isinstance(reserves, (tuple, list)) or len(reserves) != 3:
            raise ValueError("preflight requires exactly three reservations")
        if not isinstance(agent, dict) or set(agent) != {"name", "uid", "image", "node"}:
            raise ValueError("invalid agent reservation")
        required = {"name", "uid", "image", "node", "container", "gpu_uuid"}
        if any(not isinstance(row, dict) or set(row) != required for row in reserves):
            raise ValueError("invalid GPU reservation")
        names = [row["name"] for row in reserves]
        uuids = [row["gpu_uuid"] for row in reserves]
        if (len(set(names)) != 3 or len(set(uuids)) != 3
                or any(not isinstance(value, str) or not value for row in reserves for value in row.values())
                or any(row["node"] != node for row in reserves) or agent.get("node") != node
                or any(re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", row["image"]) is None for row in reserves)
                or re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", agent["image"]) is None):
            raise ValueError("invalid pinned preflight reservation")
        self.namespace, self.node, self.reserves, self.agent = namespace, node, tuple(reserves), agent
        self.transport, self.timeout_s = transport, timeout_s

    def _call(self, argv):
        result = self.transport(argv, timeout_s=self.timeout_s)
        if getattr(result, "returncode", 0) != 0:
            raise ValueError(getattr(result, "stderr", "preflight transport failure") or "preflight transport failure")
        return result.stdout

    def _json(self, argv):
        try:
            value = json.loads(self._call(argv))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid preflight JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("invalid preflight JSON")
        return value

    def _pod(self, name):
        return self._json(["kubectl", "-n", self.namespace, "get", "pod", name, "-o", "json"])

    def _agent(self, *command):
        return ["kubectl", "-n", self.namespace, "exec", self.agent["name"], "-c", "agent", "--", *command]

    @staticmethod
    def _ready(value):
        conditions = value.get("status", {}).get("conditions", []) if isinstance(value, dict) else []
        return isinstance(conditions, list) and any(
            isinstance(row, dict) and row.get("type") == "Ready" and row.get("status") == "True" for row in conditions
        )

    @staticmethod
    def _one_container(value, name):
        containers = value.get("spec", {}).get("containers", []) if isinstance(value, dict) else []
        matches = [item for item in containers if isinstance(item, dict) and item.get("name") == name]
        if len(matches) != 1:
            raise ValueError("missing required Pod container")
        return matches[0]

    @staticmethod
    def _one_status_container(value, name):
        statuses = value.get("status", {}).get("containerStatuses", []) if isinstance(value, dict) else []
        matches = [item for item in statuses if isinstance(item, dict) and item.get("name") == name]
        if len(matches) != 1:
            raise ValueError("missing required Pod container status")
        return matches[0]

    @staticmethod
    def _gpu_one(container):
        resources = container.get("resources") if isinstance(container, dict) else None
        if not isinstance(resources, dict):
            return False
        return (resources.get("requests") == {"nvidia.com/gpu": "1"}
                and resources.get("limits") == {"nvidia.com/gpu": "1"})

    @staticmethod
    def _lines(value):
        if not isinstance(value, str):
            raise ValueError("invalid GPU query output")
        values = value.splitlines()
        if not values or any(not item or item.strip() != item for item in values):
            raise ValueError("invalid GPU query output")
        return values

    def __call__(self):
        node = self._json(["kubectl", "get", "node", self.node, "-o", "json"])
        spec = node.get("spec")
        status = node.get("status")
        if (not isinstance(spec, dict) or spec.get("unschedulable", False) is not False or not isinstance(status, dict)
                or not self._ready(node) or status.get("capacity", {}).get("nvidia.com/gpu") != "4"
                or status.get("allocatable", {}).get("nvidia.com/gpu") != "4"):
            raise ValueError("node is not the pinned schedulable four-GPU host")
        expected_names = set()
        expected_uuids = set()
        for reservation in self.reserves:
            pod = self._pod(reservation["name"])
            metadata, pod_spec = pod.get("metadata"), pod.get("spec")
            if (not isinstance(metadata, dict) or metadata.get("uid") != reservation["uid"]
                    or not isinstance(pod_spec, dict) or pod_spec.get("nodeName") != self.node or not self._ready(pod)):
                raise ValueError("GPU reservation identity mismatch")
            container = self._one_container(pod, reservation["container"])
            if container.get("image") != reservation["image"] or not self._gpu_one(container):
                raise ValueError("GPU reservation workload mismatch")
            logged_uuid = self._call([
                "kubectl", "-n", self.namespace, "logs", reservation["name"], "--tail=1",
            ]).strip()
            if logged_uuid.startswith("GPU UUID: "):
                logged_uuid = logged_uuid.removeprefix("GPU UUID: ")
            if logged_uuid != reservation["gpu_uuid"]:
                raise ValueError("GPU reservation UUID mismatch")
            expected_names.add(reservation["name"])
            expected_uuids.add(reservation["gpu_uuid"])
        agent = self._pod(self.agent["name"])
        if (agent.get("metadata", {}).get("uid") != self.agent["uid"]
                or agent.get("spec", {}).get("nodeName") != self.node or not self._ready(agent)
                or self._one_status_container(agent, "agent").get("imageID") != self.agent["image"]):
            raise ValueError("agent preflight identity mismatch")
        listed = self._json(["kubectl", "get", "pods", "-A", "-o", "json"])
        items = listed.get("items")
        if not isinstance(items, list):
            raise ValueError("invalid Pod inventory")
        names = []
        for pod in items:
            if not isinstance(pod, dict):
                raise ValueError("invalid Pod inventory entry")
            metadata, pod_spec = pod.get("metadata"), pod.get("spec")
            if not isinstance(metadata, dict) or not isinstance(pod_spec, dict):
                raise ValueError("invalid Pod inventory entry")
            labels = metadata.get("labels", {})
            phase = pod.get("status", {}).get("phase")
            active = phase in {"Pending", "Running"}
            if not isinstance(labels, dict):
                raise ValueError("invalid Pod labels")
            if active and pod_spec.get("nodeName") == self.node and labels.get("poc.regolo.ai/lane") == "v2-a":
                raise ValueError("V2 workload already exists")
            if not active or pod_spec.get("nodeName") != self.node:
                continue
            gpu = 0
            containers = pod_spec.get("containers", [])
            if not isinstance(containers, list):
                raise ValueError("invalid Pod containers")
            for container in containers:
                resources = container.get("resources", {}) if isinstance(container, dict) else {}
                requests = resources.get("requests", {}) if isinstance(resources, dict) else {}
                value = requests.get("nvidia.com/gpu", "0") if isinstance(requests, dict) else "0"
                if not isinstance(value, (str, int)) or not str(value).isdigit():
                    raise ValueError("invalid GPU request")
                gpu += int(value)
            if gpu:
                names.append(metadata.get("name"))
        if set(names) != expected_names or len(names) != len(expected_names):
            raise ValueError("unexpected Pod consumer on reserved host")
        gpu_uuids = self._lines(self._call(self._agent(
            "nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"
        )))
        if len(gpu_uuids) != 4 or len(set(gpu_uuids)) != 4 or not expected_uuids.issubset(gpu_uuids):
            raise ValueError("GPU UUID inventory mismatch")
        consumer_text = self._call(self._agent(
            "nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"
        ))
        consumers = [] if not consumer_text.strip() else consumer_text.splitlines()
        if (any(not item or item.strip() != item for item in consumers)
                or len(consumers) != len(set(consumers)) or not set(consumers).issubset(expected_uuids)):
            raise ValueError("GPU consumer inventory mismatch")
        return True


def _attestation_file(path, expected_digest):
    """Read one private attestation without following a final symlink."""
    if not isinstance(expected_digest, str) or re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None:
        raise ValueError("invalid attestation digest")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(os.fspath(path), flags)
    except OSError as exc:
        raise ValueError("cannot open checkpoint attestation") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("checkpoint attestation is not a regular file")
        chunks = []
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(fd)
    body = b"".join(chunks)
    import hashlib
    if hashlib.sha256(body).hexdigest() != expected_digest:
        raise ValueError("checkpoint attestation digest mismatch")
    try:
        def reject_duplicates(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("duplicate attestation field")
                value[key] = item
            return value
        return json.loads(body.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("invalid checkpoint attestation") from exc


class ProductionCheckpointValidator:
    """Bind the signed local checkpoint description to read-only cluster facts."""

    def __init__(self, namespace, agent_pod, transport, *, timeout_s):
        if not isinstance(namespace, str) or not namespace or not isinstance(agent_pod, str) or not agent_pod:
            raise ValueError("invalid checkpoint validator target")
        self.namespace = namespace
        self.agent_pod = agent_pod
        self.transport = transport
        self.timeout_s = timeout_s

    def _call(self, argv):
        result = self.transport(argv, timeout_s=self.timeout_s)
        if getattr(result, "returncode", 0) != 0:
            raise ValueError(getattr(result, "stderr", "transport failure") or "transport failure")
        return result.stdout

    def _json(self, argv):
        try:
            value = json.loads(self._call(argv))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid remote JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("invalid remote JSON")
        return value

    def _agent(self, *command):
        return ["kubectl", "-n", self.namespace, "exec", self.agent_pod, "-c", "agent", "--", *command]

    @staticmethod
    def _exact(value, fields, description):
        if not isinstance(value, dict) or set(value) != set(fields):
            raise ValueError("invalid " + description)

    @staticmethod
    def _identifier(value, pattern, description):
        if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
            raise ValueError("invalid " + description)
        return value

    def _attestation(self, path, digest):
        value = _attestation_file(path, digest)
        self._exact(value, {"checkpoint", "agent", "pvc", "pv"}, "checkpoint attestation")
        checkpoint = value["checkpoint"]
        agent = value["agent"]
        pvc = value["pvc"]
        pv = value["pv"]
        checkpoint_fields = {
            "id", "compatibility_hash", "location", "total_size_bytes", "pages_12_size_bytes",
            "rootfs_size_bytes", "metadata_size_bytes", "manifest_sha256",
        }
        if set(checkpoint) != checkpoint_fields | {"inventory"}:
            raise ValueError("invalid checkpoint attestation")
        self._exact(agent, {"namespace", "name", "uid", "image", "node"}, "agent attestation")
        self._exact(pvc, {"name", "uid", "pv"}, "PVC attestation")
        self._exact(pv, {"uid", "local_path", "claim_uid", "node", "reclaim_policy"}, "PV attestation")
        self._identifier(checkpoint["id"], r"h-[0-9a-f]{61}", "checkpoint ID")
        self._identifier(checkpoint["compatibility_hash"], r"[0-9a-f]{64}", "compatibility hash")
        self._identifier(checkpoint["manifest_sha256"], r"[0-9a-f]{64}", "manifest digest")
        location = checkpoint["location"]
        if location != "/checkpoints/" + checkpoint["id"] + "/versions/1":
            raise ValueError("invalid checkpoint location")
        if any(not isinstance(checkpoint[name], int) or isinstance(checkpoint[name], bool) or checkpoint[name] <= 0 for name in (
            "total_size_bytes", "pages_12_size_bytes", "rootfs_size_bytes", "metadata_size_bytes",
        )):
            raise ValueError("invalid checkpoint size")
        if agent["namespace"] != self.namespace or agent["name"] != self.agent_pod:
            raise ValueError("agent identity mismatch")
        for item, description in ((agent["uid"], "agent UID"), (agent["node"], "agent node"), (pvc["name"], "PVC name"), (pvc["uid"], "PVC UID"), (pvc["pv"], "PV name"), (pv["uid"], "PV UID"), (pv["node"], "PV node")):
            self._identifier(item, r"[A-Za-z0-9][A-Za-z0-9._-]*", description)
        self._identifier(agent["image"], r"[^@\s]+@sha256:[0-9a-f]{64}", "agent image")
        local = pv["local_path"]
        if (not isinstance(local, str) or not re.fullmatch(r"/[A-Za-z0-9._/-]+", local)
                or "/../" in local or local.endswith("/..") or local == location):
            raise ValueError("invalid PV local path")
        if pv["claim_uid"] != pvc["uid"] or pv["node"] != agent["node"] or pv["reclaim_policy"] != "Retain":
            raise ValueError("PV attestation mismatch")
        return checkpoint, agent, pvc, pv

    @staticmethod
    def _single_named(items, name):
        if not isinstance(items, list):
            raise ValueError("invalid container status")
        matches = [item for item in items if isinstance(item, dict) and item.get("name") == name]
        if len(matches) != 1:
            raise ValueError("missing named container")
        return matches[0]

    @staticmethod
    def _pv_binds_node(spec, node):
        affinity = spec.get("nodeAffinity") if isinstance(spec, dict) else None
        required = affinity.get("required") if isinstance(affinity, dict) else None
        terms = required.get("nodeSelectorTerms") if isinstance(required, dict) else None
        if not isinstance(terms, list) or not terms:
            return False
        for term in terms:
            expressions = term.get("matchExpressions") if isinstance(term, dict) else None
            if not isinstance(expressions, list):
                continue
            for expression in expressions:
                if (isinstance(expression, dict) and expression.get("key") == "kubernetes.io/hostname"
                        and expression.get("operator") == "In" and expression.get("values") == [node]):
                    return True
        return False

    def _regular_size(self, path, size):
        output = self._call(self._agent("stat", "-c", "%F %s", path))
        if output != "regular file " + str(size) + "\n":
            raise ValueError("checkpoint file is not the attested regular file")

    def _inventory(self, root, value):
        fields = {"regular_file_count", "regular_file_size_bytes", "inventory_sha256"}
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("invalid checkpoint inventory")
        if (not isinstance(value["regular_file_count"], int) or isinstance(value["regular_file_count"], bool)
                or value["regular_file_count"] <= 0 or not isinstance(value["regular_file_size_bytes"], int)
                or isinstance(value["regular_file_size_bytes"], bool) or value["regular_file_size_bytes"] <= 0
                or not isinstance(value["inventory_sha256"], str) or _SHA256.fullmatch(value["inventory_sha256"]) is None):
            raise ValueError("invalid checkpoint inventory")
        listing = self._call(self._agent(
            "find", root, "-mindepth", "1", "-maxdepth", "1", "-printf", "%y|%f|%s\\n"
        ))
        rows = []
        names = set()
        for line in listing.splitlines():
            parts = line.split("|")
            if len(parts) != 3 or parts[0] != "f" or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", parts[1]) or not parts[2].isdigit():
                raise ValueError("unsafe checkpoint inventory listing")
            name, size = parts[1], int(parts[2])
            if name in names or size < 0:
                raise ValueError("duplicate checkpoint inventory entry")
            names.add(name)
            rows.append({"path": name, "size": size})
        canonical = json.dumps(sorted(rows, key=lambda row: row["path"]), sort_keys=True, separators=(",", ":")).encode()
        if (len(rows) != value["regular_file_count"] or sum(row["size"] for row in rows) != value["regular_file_size_bytes"]
                or hashlib.sha256(canonical).hexdigest() != value["inventory_sha256"]):
            raise ValueError("checkpoint inventory does not match attestation")
        return rows

    def __call__(self, attestation_path, attestation_sha256):
        checkpoint, agent, pvc, pv = self._attestation(attestation_path, attestation_sha256)
        pod = self._json(["kubectl", "-n", self.namespace, "get", "pod", self.agent_pod, "-o", "json"])
        metadata = pod.get("metadata")
        spec = pod.get("spec")
        status = pod.get("status")
        if (not isinstance(metadata, dict) or metadata.get("uid") != agent["uid"] or not isinstance(spec, dict)
                or spec.get("nodeName") != agent["node"] or not isinstance(status, dict)
                or self._single_named(status.get("containerStatuses"), "agent").get("imageID") != agent["image"]):
            raise ValueError("agent remote identity mismatch")
        remote_pvc = self._json(["kubectl", "-n", self.namespace, "get", "pvc", pvc["name"], "-o", "json"])
        if (remote_pvc.get("metadata", {}).get("uid") != pvc["uid"]
                or remote_pvc.get("spec", {}).get("volumeName") != pvc["pv"]):
            raise ValueError("PVC remote identity mismatch")
        remote_pv = self._json(["kubectl", "get", "pv", pvc["pv"], "-o", "json"])
        remote_metadata = remote_pv.get("metadata", {})
        remote_spec = remote_pv.get("spec", {})
        if (not isinstance(remote_metadata, dict) or remote_metadata.get("uid") != pv["uid"]
                or not isinstance(remote_spec, dict) or remote_spec.get("persistentVolumeReclaimPolicy") != "Retain"
                or remote_spec.get("local", {}).get("path") != pv["local_path"]
                or remote_spec.get("claimRef", {}).get("uid") != pvc["uid"]
                or not self._pv_binds_node(remote_spec, agent["node"])):
            raise ValueError("PV remote identity mismatch")
        root = checkpoint["location"]
        self._inventory(root, checkpoint["inventory"])
        self._regular_size(root + "/pages-12.img", checkpoint["pages_12_size_bytes"])
        self._regular_size(root + "/rootfs-diff.tar", checkpoint["rootfs_size_bytes"])
        self._regular_size(root + "/manifest.yaml", checkpoint["metadata_size_bytes"])
        total = self._call(self._agent("du", "-sb", root))
        if total != str(checkpoint["total_size_bytes"]) + "\t" + root + "\n":
            raise ValueError("checkpoint total size mismatch")
        manifest = self._call(self._agent("cat", root + "/manifest.yaml"))
        matches = re.findall(
            r"(?m)^checkpointId:[ \t]*" + re.escape(checkpoint["id"]) + r"[ \t]*$",
            manifest if isinstance(manifest, str) else "",
        )
        if len(matches) != 1:
            raise ValueError("checkpoint manifest identity mismatch")
        hash_line = self._call(self._agent("sha256sum", root + "/manifest.yaml"))
        expected_line = checkpoint["manifest_sha256"] + "  " + root + "/manifest.yaml\n"
        if hash_line != expected_line:
            raise ValueError("checkpoint manifest digest mismatch")
        return True


class ProductionCollector:
    def __init__(self, namespace, agent_pod, artifact_dir, checkpoint_attestation, transport, *, timeout_s):
        self.namespace = namespace
        self.agent_pod = agent_pod
        self.artifact_dir = pathlib.Path(artifact_dir)
        self.attestation = checkpoint_attestation
        self.transport = transport
        self.timeout_s = timeout_s
        self._prepared = {}

    def _call(self, argv):
        result = self.transport(argv, timeout_s=self.timeout_s)
        if getattr(result, "returncode", 0) != 0:
            raise ValueError(getattr(result, "stderr", "transport failure") or "transport failure")
        return result.stdout

    def _pod_command(self, pod_name):
        return ["kubectl", "-n", self.namespace, "get", "pod", pod_name, "-o", "json"]

    def _agent(self, *command):
        return ["kubectl", "-n", self.namespace, "exec", self.agent_pod, "-c", "agent", "--", *command]

    def _workload(self, pod_name, *command):
        return ["kubectl", "-n", self.namespace, "exec", pod_name, "-c", "server", "--", *command]

    @staticmethod
    def _attestation(value):
        fields = {
            "checkpoint_id", "compatibility_hash", "checkpoint_size_bytes",
            "pages_12_size_bytes", "rootfs_size_bytes", "metadata_size_bytes",
        }
        if not isinstance(value, dict) or set(value) not in (fields, fields | {"checkpoint_inventory"}):
            raise ValueError("invalid checkpoint attestation")
        if not re.fullmatch(r"h-[0-9a-f]{61}", value["checkpoint_id"]) or not re.fullmatch(r"[0-9a-f]{64}", value["compatibility_hash"]):
            raise ValueError("invalid checkpoint identity")
        if any(not isinstance(value[field], int) or isinstance(value[field], bool) or value[field] <= 0 for field in fields - {"checkpoint_id", "compatibility_hash"}):
            raise ValueError("invalid checkpoint sizes")
        if "checkpoint_inventory" in value:
            inventory = value["checkpoint_inventory"]
            inventory_fields = {"regular_file_count", "regular_file_size_bytes", "inventory_sha256"}
            if (
                not isinstance(inventory, dict) or set(inventory) != inventory_fields
                or not isinstance(inventory["regular_file_count"], int)
                or isinstance(inventory["regular_file_count"], bool)
                or inventory["regular_file_count"] <= 0
                or not isinstance(inventory["regular_file_size_bytes"], int)
                or isinstance(inventory["regular_file_size_bytes"], bool)
                or inventory["regular_file_size_bytes"] <= 0
                or not isinstance(inventory["inventory_sha256"], str)
                or _SHA256.fullmatch(inventory["inventory_sha256"]) is None
            ):
                raise ValueError("invalid checkpoint inventory")

    def _write(self, relative, value):
        path = self.artifact_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            body = value if isinstance(value, bytes) else str(value).encode()
            if os.write(fd, body) != len(body):
                raise OSError("short evidence write")
            os.fsync(fd)
        finally:
            os.close(fd)
        return relative

    def prepare(self, run, mode, pod_name):
        self._attestation(self.attestation)
        self._prepared[run["run_id"]] = {
            "since": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "host_before": self._host_snapshot(),
        }

    def _host_snapshot(self):
        return {
            "meminfo": self._call(self._agent("cat", "/host/proc/meminfo")),
            "psi_cpu": self._call(self._agent("cat", "/host/proc/pressure/cpu")),
            "psi_io": self._call(self._agent("cat", "/host/proc/pressure/io")),
            "psi_memory": self._call(self._agent("cat", "/host/proc/pressure/memory")),
            "cgroup_io_stat": self._call(self._agent("cat", "/sys/fs/cgroup/io.stat")),
            "diskstats": self._call(self._agent("cat", "/host/proc/diskstats")),
            "cpu": self._call(self._agent("cat", "/host/proc/stat")),
        }

    def _ready_pod(self, pod_name):
        deadline = time.monotonic() + self.timeout_s
        uid = None
        while True:
            pod = _json(self.transport(self._pod_command(pod_name), timeout_s=self.timeout_s))
            current = pod.get("metadata", {}).get("uid")
            if not isinstance(current, str) or not current:
                raise ValueError("pod lacks UID")
            if uid is None:
                uid = current
            elif current != uid:
                raise ValueError("pod UID changed while collecting")
            try:
                _condition(pod, "Ready")
                return pod
            except ValueError:
                if time.monotonic() >= deadline:
                    raise ValueError("pod did not become Ready")
                time.sleep(0.01)

    def _restore_completed_pod(self, pod_name, pod_uid):
        deadline = time.monotonic() + self.timeout_s
        last_poll = None
        while True:
            pod = _json(self.transport(self._pod_command(pod_name), timeout_s=self.timeout_s))
            metadata = pod.get("metadata") if isinstance(pod, dict) else None
            if not isinstance(metadata, dict) or metadata.get("uid") != pod_uid:
                raise ValueError("Pod UID changed while waiting for restore completion")
            annotations = metadata.get("annotations", {})
            if not isinstance(annotations, dict):
                raise ValueError("invalid restore annotations")
            status = annotations.get("nvidia.com/snapshot-restore-status.server")
            if status == "completed":
                return pod
            if status is not None and not isinstance(status, str):
                raise ValueError("invalid restore completion annotation")
            now = time.monotonic()
            if now >= deadline or (last_poll is not None and now <= last_poll):
                raise ValueError("restore completion annotation did not arrive before deadline")
            last_poll = now
            time.sleep(0.01)

    def __call__(self, run, mode, pod_name):
        prepared = self._prepared.get(run.get("run_id"))
        if prepared is None:
            raise ValueError("collector was not prepared")
        pod = self._ready_pod(pod_name)
        metadata = pod["metadata"]
        created = _epoch(metadata["creationTimestamp"])
        scheduled = _condition(pod, "PodScheduled")
        ready = _condition(pod, "Ready")
        logs_command = [
            "kubectl", "-n", self.namespace, "logs", self.agent_pod, "-c", "agent",
            "--since-time=" + prepared["since"],
        ]
        logs = self._call(logs_command)
        summary = storage_after = None
        if mode == "restore":
            deadline = time.monotonic() + self.timeout_s
            container_id = _server_container_id(pod)
            last_poll = None
            while True:
                if re.search(r"NVRM: Xid|OOMKilled|I/O error", logs, re.I):
                    raise ValueError("terminal agent fault")
                try:
                    start, summary, restore = _restore_log(
                        logs, self.namespace + "/" + pod_name, container_id, created, self.attestation["checkpoint_id"]
                    )
                    break
                except _RestorePending:
                    now = time.monotonic()
                    if now >= deadline or (last_poll is not None and now <= last_poll):
                        raise ValueError("restore log pair did not arrive before deadline")
                    last_poll = now
                    time.sleep(0.01)
                    logs = self._call(logs_command)
            pod = self._restore_completed_pod(pod_name, metadata["uid"])
            metadata = pod["metadata"]
            created = _epoch(metadata["creationTimestamp"])
            scheduled = _condition(pod, "PodScheduled")
            ready = _condition(pod, "Ready")
            storage_after = self._host_snapshot()
        else:
            if re.search(r"NVRM: Xid|OOMKilled|I/O error", logs, re.I):
                raise ValueError("terminal agent fault")
            start = summary = restore = None
        events = self._call([
            "kubectl", "-n", self.namespace, "get", "events", "--field-selector",
            "involvedObject.uid=" + metadata["uid"], "-o", "json",
        ])
        if mode == "restore":
            try:
                _restore_events(json.loads(events), self.namespace, pod_name, metadata["uid"], self.attestation["checkpoint_id"])
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError("invalid restore events") from exc
        probe = (
            "import json,re,time,urllib.request\n"
            "base='http://localhost:8000'\n"
            "with urllib.request.urlopen(base+'/health',timeout=5): pass\n"
            "payload=json.dumps({'model':'openai/gpt-oss-20b','prompt':'The answer to 1+1 is','max_tokens':128,'temperature':0,'stream':True}).encode()\n"
            "request=urllib.request.Request(base+'/v1/completions',data=payload,headers={'Content-Type':'application/json'},method='POST')\n"
            "parts=[]; completion_tokens=None; first_wall=None; first_mono=None\n"
            "with urllib.request.urlopen(request,timeout=30) as stream:\n"
            "    http_200=time.time()\n"
            "    for raw in stream:\n"
            "        line=raw.decode('utf-8').strip()\n"
            "        if not line.startswith('data:'): continue\n"
            "        data=line[5:].strip()\n"
            "        if data=='[DONE]': break\n"
            "        event=json.loads(data); choices=event.get('choices')\n"
            "        if not isinstance(choices,list): raise RuntimeError('missing choices')\n"
            "        usage=event.get('usage',{})\n"
            "        if isinstance(usage,dict) and isinstance(usage.get('completion_tokens'),int): completion_tokens=usage['completion_tokens']\n"
            "        for choice in choices:\n"
            "            text=choice.get('text') if isinstance(choice,dict) else None\n"
            "            if isinstance(text,str) and text:\n"
            "                if first_wall is None: first_wall=time.time(); first_mono=time.monotonic()\n"
            "                parts.append(text)\n"
            "if first_wall is None: raise RuntimeError('no first token')\n"
            "response=''.join(parts)\n"
            "if not re.match(r'\\s*2\\b',response): raise RuntimeError('invalid completion')\n"
            "with urllib.request.urlopen(base+'/metrics',timeout=5) as metrics_response: metrics=metrics_response.read().decode('utf-8')\n"
            "def gauge(name):\n"
            "    match=re.search(r'(?m)^'+re.escape(name)+r'(?:\\{[^}]*\\})?\\s+([0-9]+(?:\\.[0-9]+)?)\\s*$',metrics)\n"
            "    if not match: raise RuntimeError('missing '+name)\n"
            "    return int(float(match.group(1)))\n"
            "tokens=completion_tokens if completion_tokens is not None else len(parts)\n"
            "if not isinstance(tokens,int) or tokens<=0: raise RuntimeError('missing completion tokens')\n"
            "elapsed=time.monotonic()-first_mono\n"
            "if elapsed<=0: raise RuntimeError('nonpositive generation interval')\n"
            "print(json.dumps({'response':response,'http_200_epoch_s':http_200,'first_token_epoch_s':first_wall,'tokens_per_second':tokens/elapsed,'running':gauge('vllm:num_requests_running'),'waiting':gauge('vllm:num_requests_waiting')},sort_keys=True))"
        )
        response = _probe_observation(_json(self.transport(
            self._workload(pod_name, "python3", "-c", probe), timeout_s=self.timeout_s
        )))
        final = self._host_snapshot()
        host = final
        host["cpu_before"] = prepared["host_before"]["cpu"]
        host["cpu_after"] = host.pop("cpu")
        host["gpu"] = self._call(self._workload(
            pod_name, "nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"
        ))
        memory = _harness.parse_meminfo(host["meminfo"])
        before_memory = _harness.parse_meminfo(prepared["host_before"]["meminfo"])
        if mode == "restore":
            storage_memory = _harness.parse_meminfo(storage_after["meminfo"])
            cgroup_delta = CacheAdvisor._counter_delta(
                _harness.parse_io_stat(prepared["host_before"]["cgroup_io_stat"]),
                _harness.parse_io_stat(storage_after["cgroup_io_stat"]),
            )
            disk_delta = CacheAdvisor._counter_delta(
                _harness.parse_diskstats(prepared["host_before"]["diskstats"]),
                _harness.parse_diskstats(storage_after["diskstats"]), ("dm-0", "loop6", "sda"),
            )
            storage_read = cgroup_delta.get("253:0", {}).get("rbytes")
            if not isinstance(storage_read, int) or storage_read <= 0 or summary <= start:
                raise ValueError("invalid checkpoint storage evidence")
        else:
            storage_memory = None
            cgroup_delta = disk_delta = storage_read = None
        metrics = {
            "pod_to_scheduled_s": scheduled - created,
            "pod_to_restore_start_s": max(0.0, start - created) if mode == "restore" else None,
            "criu_restore_s": _seconds(restore["criu_restore_duration"]) if mode == "restore" else None,
            "cuda_restore_s": _seconds(restore["cuda_duration"]) if mode == "restore" else None,
            "ready_s": ready - created,
            "http_200_s": float(response["http_200_epoch_s"]) - created,
            "first_token_s": float(response["first_token_epoch_s"]) - created,
            "cgroup_io_stat": cgroup_delta if mode == "restore" else _harness.parse_io_stat(host["cgroup_io_stat"]),
            "diskstats": disk_delta if mode == "restore" else _harness.parse_diskstats(host["diskstats"]),
            "node_page_cache_bytes": memory["page_cache_bytes"],
            "node_memory_available_bytes": memory["mem_available_bytes"],
            "psi_cpu": _harness.parse_psi(host["psi_cpu"]),
            "psi_io": _harness.parse_psi(host["psi_io"]),
            "psi_memory": _harness.parse_psi(host["psi_memory"]),
            "node_cpu_utilization": _harness.cpu_utilization(host["cpu_before"], host["cpu_after"]),
            "gpu_memory_mib": _harness.parse_gpu_memory_mib(host["gpu"]),
            "checkpoint_size_bytes": self.attestation["checkpoint_size_bytes"],
            "pages_12_size_bytes": self.attestation["pages_12_size_bytes"],
            "rootfs_size_bytes": self.attestation["rootfs_size_bytes"],
            "metadata_size_bytes": self.attestation["metadata_size_bytes"],
            "prepare_s": None,
            "sleep_s": None,
            "wake_s": None,
            "admission_closed": True,
            "harness_inflight": 0,
            "vllm_running": response.get("running"),
            "vllm_waiting": response.get("waiting"),
            "tokens_per_second": float(response["tokens_per_second"]),
            "token_after_restore_summary_s": float(response["first_token_epoch_s"]) - summary if mode == "restore" else None,
            "checkpoint_storage_read_bytes": storage_read,
            "checkpoint_storage_read_throughput_bytes_s": storage_read / (summary - start) if mode == "restore" else None,
            "node_page_cache_delta_bytes": (storage_memory["page_cache_bytes"] - before_memory["page_cache_bytes"] if mode == "restore" else memory["page_cache_bytes"] - before_memory["page_cache_bytes"]),
            "node_memory_available_delta_bytes": (storage_memory["mem_available_bytes"] - before_memory["mem_available_bytes"] if mode == "restore" else memory["mem_available_bytes"] - before_memory["mem_available_bytes"]),
        }
        if any(not isinstance(metrics[field], (int, float)) or isinstance(metrics[field], bool) or metrics[field] < 0 for field in ("pod_to_scheduled_s", "ready_s", "http_200_s", "first_token_s", "tokens_per_second")):
            raise ValueError("invalid timing evidence")
        if mode == "restore" and (metrics["pod_to_restore_start_s"] < 0 or metrics["criu_restore_s"] < 0 or metrics["cuda_restore_s"] < 0):
            raise ValueError("invalid restore timing")
        if (
            metrics["first_token_s"] < metrics["http_200_s"]
            or metrics["http_200_s"] < metrics["ready_s"]
            or metrics["ready_s"] < metrics["pod_to_scheduled_s"]
            or (mode == "restore" and (
                metrics["token_after_restore_summary_s"] < 0
                or metrics["checkpoint_storage_read_throughput_bytes_s"] <= 0
            ))
        ):
            raise ValueError("out-of-order timing evidence")
        base = "raw/" + run["run_id"]
        return {
            "metrics": metrics,
            "pod_uid": metadata["uid"],
            "pod_creation_epoch_s": created,
            "valid_response": True,
            "restore_success": mode != "restore" or metadata["annotations"]["nvidia.com/snapshot-restore-status.server"] == "completed",
            "raw_events_ref": self._write(base + ".events.json", events),
            "raw_logs_ref": self._write(base + ".logs.jsonl", logs),
            "raw_telemetry_ref": self._write(base + ".telemetry.json", json.dumps({"host_before": prepared["host_before"], "storage_after": storage_after, "final": final}, sort_keys=True)),
            "raw_response_ref": self._write(base + ".response.json", json.dumps(response, sort_keys=True)),
        }


class CacheAdvisor:
    def __init__(self, namespace, agent_pod, allow_root, transport, *, timeout_s):
        self.namespace = namespace
        self.agent_pod = agent_pod
        self.root = pathlib.PurePosixPath(allow_root)
        if not self.root.is_absolute():
            raise ValueError("allow root must be absolute")
        self.transport = transport
        self.timeout_s = timeout_s

    def _agent(self, *command):
        return ["kubectl", "-n", self.namespace, "exec", self.agent_pod, "-c", "agent", "--", *command]

    def _read(self, path):
        result = self.transport(self._agent("cat", path), timeout_s=self.timeout_s)
        if getattr(result, "returncode", 0) != 0:
            raise ValueError(getattr(result, "stderr", "telemetry read failed") or "telemetry read failed")
        return result.stdout

    def _storage_snapshot(self):
        memory = _harness.parse_meminfo(self._read("/host/proc/meminfo"))
        psi = _harness.parse_psi(self._read("/host/proc/pressure/io"))
        diskstats = _harness.parse_diskstats(self._read("/host/proc/diskstats"))
        if not {"dm-0", "loop6", "sda"}.issubset(diskstats):
            raise ValueError("missing required storage device")
        return {
            "memory": memory,
            "psi": psi,
            "diskstats": diskstats,
            "cgroup": _harness.parse_io_stat(self._read("/sys/fs/cgroup/io.stat")),
        }

    @staticmethod
    def _counter_delta(before, after, required=()):
        if not set(required).issubset(before) or not set(required).issubset(after):
            raise ValueError("missing required counter")
        output = {}
        for name in required or before:
            left, right = before[name], after.get(name)
            if not isinstance(left, dict) or not isinstance(right, dict) or set(left) != set(right):
                raise ValueError("inconsistent counter schema")
            delta = {}
            for key, value in left.items():
                later = right[key]
                if not isinstance(value, int) or not isinstance(later, int) or later < value:
                    raise ValueError("counter regressed")
                delta[key] = later - value
            output[name] = delta
        return output

    @staticmethod
    def _write_storage_artifact(directory, relative, value):
        directory = pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        info = os.lstat(directory)
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise ValueError("storage artifact directory is unsafe")
        path = directory / relative
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags, 0o600)
        except FileExistsError as exc:
            raise ValueError("refusing to overwrite storage evidence") from exc
        try:
            encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
            if os.write(fd, encoded) != len(encoded):
                raise OSError("short storage evidence write")
            os.fsync(fd)
        finally:
            os.close(fd)
        directory_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return relative

    def _files(self, paths):
        result = []
        for path in paths:
            path = pathlib.PurePosixPath(path)
            if not path.is_absolute() or ".." in path.parts:
                raise ValueError("cache candidate must be an absolute normalized file")
            try:
                path.relative_to(self.root)
            except ValueError as exc:
                raise ValueError("cache candidate is outside allow root") from exc
            text = str(path)
            result_check = self.transport(
                ["kubectl", "-n", self.namespace, "exec", self.agent_pod, "-c", "agent", "--", "stat", "-c", "%F %s", text],
                timeout_s=self.timeout_s,
            )
            if getattr(result_check, "returncode", 0) != 0 or not re.fullmatch(r"regular file [0-9]+\n?", result_check.stdout):
                raise ValueError("agent did not attest a regular cache file")
            result.append(text)
        return result

    def _inventory_paths(self, inventory):
        fields = {"regular_file_count", "regular_file_size_bytes", "inventory_sha256"}
        if not isinstance(inventory, dict) or set(inventory) != fields:
            raise ValueError("invalid checkpoint inventory")
        if (not isinstance(inventory["regular_file_count"], int) or isinstance(inventory["regular_file_count"], bool)
                or not isinstance(inventory["regular_file_size_bytes"], int) or isinstance(inventory["regular_file_size_bytes"], bool)
                or inventory["regular_file_count"] <= 0 or inventory["regular_file_size_bytes"] <= 0
                or not isinstance(inventory["inventory_sha256"], str) or _SHA256.fullmatch(inventory["inventory_sha256"]) is None):
            raise ValueError("invalid checkpoint inventory")
        listing = self._read_listing()
        rows, seen = [], set()
        for line in listing.splitlines():
            parts = line.split("|")
            if len(parts) != 3 or parts[0] != "f" or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", parts[1]) or not parts[2].isdigit():
                raise ValueError("unsafe checkpoint inventory listing")
            if parts[1] in seen:
                raise ValueError("duplicate checkpoint inventory entry")
            seen.add(parts[1])
            rows.append({"path": parts[1], "size": int(parts[2])})
        canonical = json.dumps(sorted(rows, key=lambda row: row["path"]), sort_keys=True, separators=(",", ":")).encode()
        if (len(rows) != inventory["regular_file_count"] or sum(row["size"] for row in rows) != inventory["regular_file_size_bytes"]
                or hashlib.sha256(canonical).hexdigest() != inventory["inventory_sha256"]):
            raise ValueError("checkpoint inventory does not match")
        return [str(self.root / row["path"]) for row in sorted(rows, key=lambda row: row["path"])]

    def _read_listing(self):
        result = self.transport(self._agent("find", str(self.root), "-mindepth", "1", "-maxdepth", "1", "-printf", "%y|%f|%s\\n"), timeout_s=self.timeout_s)
        if getattr(result, "returncode", 0) != 0:
            raise ValueError(getattr(result, "stderr", "inventory listing failed") or "inventory listing failed")
        return result.stdout

    def advise_inventory(self, inventory):
        files = self._files(self._inventory_paths(inventory))
        for path in files:
            self._dd(path, advice=True)

    def _dd(self, path, *, direct=False, count=0, advice=False):
        command = ["kubectl", "-n", self.namespace, "exec", self.agent_pod, "-c", "agent", "--", "dd", f"if={path}", "of=/dev/null"]
        if advice:
            command.extend(["iflag=nocache", "count=0"])
        elif direct:
            command.extend(["bs=4M", "iflag=direct,count_bytes", f"count={count}"])
        else:
            command.extend(["iflag=count_bytes", f"count={count}"])
        command.append("status=none")
        result = self.transport(command, timeout_s=self.timeout_s)
        if getattr(result, "returncode", 0) != 0:
            raise ValueError(getattr(result, "stderr", "dd failed") or "dd failed")
        return result.stdout

    def advise(self, paths):
        for path in self._files(paths):
            self._dd(path, advice=True)

    def characterize_storage(self, paths, *, max_bytes, max_reads, artifact_dir):
        if not isinstance(max_bytes, int) or not isinstance(max_reads, int) or max_bytes < 0 or max_reads < 0:
            raise ValueError("invalid storage bounds")
        files = self._files(paths)
        if not files or max_reads < 3:
            raise ValueError("three bounded reads require an allowlisted file")
        self._dd(files[0], advice=True)
        reads = []
        for mode in ("buffered-first", "buffered-repeat", "direct"):
            before = self._storage_snapshot()
            started = time.monotonic()
            self._dd(files[0], direct=mode == "direct", count=max_bytes)
            wall_s = max(time.monotonic() - started, 1e-9)
            after = self._storage_snapshot()
            if set(after["psi"]) != set(before["psi"]):
                raise ValueError("inconsistent I/O PSI")
            psi_delta = {}
            for name in after["psi"]:
                later, prior = after["psi"][name]["total"], before["psi"][name]["total"]
                if later < prior:
                    raise ValueError("I/O PSI counter regressed")
                psi_delta[name] = later - prior
            reads.append({
                "mode": mode, "bytes": max_bytes, "wall_s": wall_s,
                "throughput_bytes_s": max_bytes / wall_s,
                "page_cache_delta_bytes": after["memory"]["page_cache_bytes"] - before["memory"]["page_cache_bytes"],
                "mem_available_delta_bytes": after["memory"]["mem_available_bytes"] - before["memory"]["mem_available_bytes"],
                "psi_io_total_delta": psi_delta,
                "diskstats_delta": self._counter_delta(before["diskstats"], after["diskstats"], ("dm-0", "loop6", "sda")),
                "cgroup_io_delta": self._counter_delta(before["cgroup"], after["cgroup"]),
            })
        raw_ref = "storage-characterization.json"
        self._write_storage_artifact(artifact_dir, raw_ref, reads)
        return {"reads": reads, "raw_ref": raw_ref}
