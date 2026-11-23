# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Temporary observation for the requested UCX comparison; remove after CI.

Loaded only through the comparison job's PYTHONPATH. Wrap the installed NIXL
Python API when it is normally imported, without importing CUDA/NIXL early.
Both variants have the same instrumentation. Logging can affect timing;
unchanged polling states are sampled at five-second intervals.
"""

import functools
import hashlib
import importlib.abc
import importlib.machinery
import inspect
import json
import os
import sys
import time
from pathlib import Path

METHODS = (
    "prep_xfer_dlist",
    "make_prepped_xfer",
    "initialize_xfer",
    "transfer",
    "check_xfer_state",
    "release_xfer_handle",
    "get_new_notifs",
    "update_notifs",
    "check_remote_xfer_done",
    "send_notif",
)
POLL_METHODS = {
    "check_xfer_state",
    "check_remote_xfer_done",
    "get_new_notifs",
    "update_notifs",
}
last_poll = {}


def emit(event, **fields):
    print(
        "NURSERY_NIXL "
        + json.dumps(
            {
                "event": event,
                "pid": os.getpid(),
                "time_ns": time.time_ns(),
                "attempt": os.environ["NURSERY_UCX_ATTEMPT"],
                **fields,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )


def handle_id(handle):
    # Installed API handles wrap the real opaque integer in _handle.
    return str(handle._handle)


def tag(value):
    if isinstance(value, bytes):
        # Completion tags contain request IDs, not memory descriptors/payloads.
        return value[:256].decode("utf-8", errors="backslashreplace")
    return str(value)[:256]


def details(arguments):
    fields = {}
    for name, value in arguments.items():
        if name in {"handle", "local_xfer_side", "remote_xfer_side"}:
            fields[name] = handle_id(value)
        elif name in {
            "xfer_list",
            "local_descs",
            "remote_descs",
            "local_indices",
            "remote_indices",
        }:
            # Some native descriptor lists expose descCount() rather than len().
            fields[name + "_count"] = (
                len(value) if hasattr(value, "__len__") else value.descCount()
            )
        elif name in {
            "operation",
            "agent_name",
            "remote_agent",
            "remote_agent_name",
            "mem_type",
        }:
            fields[name] = value
        elif name in {"notif_msg", "lookup_tag"}:
            fields[name] = tag(value)
    return fields


def wrap(method, original):
    signature = inspect.signature(original)

    @functools.wraps(original)
    def observed(self, *args, **kwargs):
        arguments = signature.bind(self, *args, **kwargs).arguments
        fields = details(arguments)
        if method not in POLL_METHODS:
            emit(method + "_call", agent_object=hex(id(self)), **fields)
        try:
            result = original(self, *args, **kwargs)
        except Exception as exc:
            emit(method + "_error", error_type=type(exc).__name__, **fields)
            raise
        if method in {"prep_xfer_dlist", "make_prepped_xfer", "initialize_xfer"}:
            fields["result_handle"] = handle_id(result)
        elif isinstance(result, dict):
            fields["notifications"] = {
                str(agent): [tag(message) for message in messages]
                for agent, messages in result.items()
            }
        elif isinstance(result, (str, bool)):
            fields["result"] = result
        if method in POLL_METHODS:
            key = (id(self), method, fields.get("handle"), fields.get("lookup_tag"))
            value = json.dumps(fields, sort_keys=True)
            now = time.monotonic()
            previous = last_poll.get(key)
            # Always print new states and nonempty notifications.
            if (
                previous
                and previous[0] == value
                and now - previous[1] < 5
                and not result
            ):
                return result
            if (
                previous
                and previous[0] == value
                and now - previous[1] < 5
                and isinstance(result, str)
            ):
                return result
            last_poll[key] = (value, now)
        emit(method + "_return", agent_object=hex(id(self)), **fields)
        if method == "release_xfer_handle":
            last_poll.pop((id(self), "check_xfer_state", fields["handle"], None), None)
        return result

    return observed


class TraceLoader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        agent_class = module.nixl_agent
        available = [method for method in METHODS if hasattr(agent_class, method)]
        emit(
            "api_loaded",
            module=module.__name__,
            source=module.__file__,
            source_sha256=hashlib.sha256(
                Path(module.__file__).read_bytes()
            ).hexdigest(),
            methods=available,
            cuda_ipc_get=os.environ.get("UCX_CUDA_IPC_ENABLE_GET_ZCOPY", "unset"),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        )
        for method in available:
            setattr(agent_class, method, wrap(method, getattr(agent_class, method)))


class TraceFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("nixl.") and fullname.endswith("._api"):
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = TraceLoader(spec.loader)
                return spec
        return None


sys.meta_path.insert(0, TraceFinder())
