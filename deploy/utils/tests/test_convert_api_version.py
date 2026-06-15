# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for the apiVersion conversion utility (convert_api_version)."""

import json
import shutil
import subprocess as _sp
from pathlib import Path as _Path

import pytest

from deploy.utils import convert_api_version as c

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]


def c_test_normalize(value):
    """Recursively drop None values and empty dict/list members so two
    manifests can be compared for semantic (not cosmetic) equality."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            nv = c_test_normalize(v)
            if nv is None or nv == {} or nv == []:
                continue
            out[k] = nv
        return out
    if isinstance(value, list):
        return [c_test_normalize(v) for v in value]
    return value


def test_load_docs_parses_multiple_documents_and_drops_empty():
    """Load docs parses multiple documents and drops empty."""
    text = """\
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: a
---
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: cm
"""
    docs = c.load_docs(text)
    assert len(docs) == 2
    assert docs[0]["metadata"]["name"] == "a"
    assert docs[1]["kind"] == "ConfigMap"


def test_load_docs_keeps_date_like_scalars_as_strings():
    """Date-like scalars stay strings so the doc remains JSON-serializable.

    Regression: PyYAML's default resolver turns these into datetime/date
    objects, which json.dump then rejects when building the webhook request.
    """
    text = """\
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: a
  creationTimestamp: 2026-06-15T00:00:00Z
spec:
  someDate: 2026-06-15
"""
    doc = c.load_docs(text)[0]
    assert doc["metadata"]["creationTimestamp"] == "2026-06-15T00:00:00Z"
    assert doc["spec"]["someDate"] == "2026-06-15"
    # the whole doc must stay JSON-serializable (build_conversion_review + json.dump)
    json.dumps(c.build_conversion_review([doc], "nvidia.com/v1beta1", "uid"))


def test_is_convertible_only_true_for_nvidia_group():
    """Is convertible only true for nvidia group."""
    assert c.is_convertible({"apiVersion": "nvidia.com/v1alpha1", "kind": "X"}) is True
    assert c.is_convertible({"apiVersion": "nvidia.com/v1beta1", "kind": "X"}) is True
    assert c.is_convertible({"apiVersion": "v1", "kind": "ConfigMap"}) is False
    assert c.is_convertible({"apiVersion": "apps/v1", "kind": "Deployment"}) is False
    assert c.is_convertible({"apiVersion": "nvidia.com"}) is False  # no version part
    assert c.is_convertible({}) is False


def test_build_conversion_review_shape():
    """Build conversion review shape."""
    objs = [{"apiVersion": "nvidia.com/v1alpha1", "kind": "DynamoGraphDeployment"}]
    review = c.build_conversion_review(objs, "nvidia.com/v1beta1", "uid-123")
    assert review["apiVersion"] == "apiextensions.k8s.io/v1"
    assert review["kind"] == "ConversionReview"
    assert review["request"]["uid"] == "uid-123"
    assert review["request"]["desiredAPIVersion"] == "nvidia.com/v1beta1"
    assert review["request"]["objects"] == objs


def _response(uid="u", status="Success", objects=None, message=None):
    """Build a ConversionReview response payload for tests."""
    result = {"status": status}
    if message is not None:
        result["message"] = message
    return {
        "response": {"uid": uid, "result": result, "convertedObjects": objects or []}
    }


def test_parse_conversion_response_returns_converted_objects():
    """Parse conversion response returns converted objects."""
    obj = {"apiVersion": "nvidia.com/v1beta1", "kind": "DynamoGraphDeployment"}
    out = c.parse_conversion_response(_response(uid="u", objects=[obj]), "u")
    assert out == [obj]


def test_parse_conversion_response_raises_on_failed_status():
    """Parse conversion response raises on failed status."""
    resp = _response(status="Failed", message="bad spec")
    with pytest.raises(c.ConversionError, match="bad spec"):
        c.parse_conversion_response(resp, "u")


def test_parse_conversion_response_raises_on_uid_mismatch():
    """Parse conversion response raises on uid mismatch."""
    with pytest.raises(c.ConversionError, match="uid"):
        c.parse_conversion_response(_response(uid="other"), "u")


def test_clean_for_authoring_strips_server_fields_preserves_annotations():
    """Clean for authoring strips server fields preserves annotations."""
    obj = {
        "metadata": {
            "name": "m",
            "uid": "x",
            "resourceVersion": "123",
            "generation": 4,
            "creationTimestamp": "2026-01-01T00:00:00Z",
            "managedFields": [{"a": 1}],
            "annotations": {
                "keep": "yes",
                "nvidia.com/dgd-spec": "blob",
                "kubectl.kubernetes.io/last-applied-configuration": "blob",
            },
        },
        "spec": {"x": 1},
        "status": {"state": "successful"},
        "kind": "DynamoGraphDeployment",
        "apiVersion": "nvidia.com/v1beta1",
    }
    out = c.clean_for_authoring(obj)
    assert "status" not in out
    md = out["metadata"]
    for f in (
        "uid",
        "resourceVersion",
        "generation",
        "creationTimestamp",
        "managedFields",
    ):
        assert f not in md
    # annotations are left untouched (part of the conversion contract)
    assert md["annotations"] == {
        "keep": "yes",
        "nvidia.com/dgd-spec": "blob",
        "kubectl.kubernetes.io/last-applied-configuration": "blob",
    }
    assert list(out.keys())[:4] == ["apiVersion", "kind", "metadata", "spec"]
    assert "status" in obj


def _echo_webhook_fn_factory(converted_by_kind):
    """Return a fake webhook_fn that echoes the request uid and returns
    a canned converted object list keyed by kind."""

    def webhook_fn(review, kind):
        """Echo the request uid and return the canned converted objects for the kind."""
        return {
            "response": {
                "uid": review["request"]["uid"],
                "result": {"status": "Success"},
                "convertedObjects": converted_by_kind[kind],
            }
        }

    return webhook_fn


def test_convert_docs_passes_through_non_nvidia_docs_unchanged():
    """Convert docs passes through non nvidia docs unchanged."""
    cm = {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "cm"}}
    out = c.convert_docs(
        [cm], target="nvidia.com/v1beta1", webhook_fn=lambda review, kind: None
    )
    assert out == [cm]


def test_convert_docs_skips_docs_already_at_target():
    """Convert docs skips docs already at target."""
    doc = {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "a"},
    }
    out = c.convert_docs(
        [doc], target="nvidia.com/v1beta1", webhook_fn=lambda review, kind: None
    )
    assert out == [doc]


def test_convert_docs_converts_and_cleans_preserving_order():
    """Convert docs converts and cleans preserving order."""
    alpha = {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "a"},
        "spec": {"services": {}},
    }
    cm = {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "cm"}}
    beta = {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "a", "uid": "should-be-stripped"},
        "spec": {"components": []},
        "status": {"state": "x"},
    }
    fn = _echo_webhook_fn_factory({"DynamoGraphDeployment": [beta]})
    out = c.convert_docs([cm, alpha], target="nvidia.com/v1beta1", webhook_fn=fn)
    assert out[0] == cm
    assert out[1]["apiVersion"] == "nvidia.com/v1beta1"
    assert "uid" not in out[1]["metadata"]
    assert "status" not in out[1]


def test_convert_docs_preserves_all_annotations():
    """Convert docs preserves all annotations."""
    alpha = {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "a"},
        "spec": {},
    }
    annotations = {
        "nvidia.com/dgd-spec": "blob",
        "nvidia.com/dgd-status": "blob",
    }
    beta = {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "a", "annotations": dict(annotations)},
        "spec": {},
    }
    fn = _echo_webhook_fn_factory({"DynamoGraphDeployment": [beta]})
    out = c.convert_docs([alpha], target="nvidia.com/v1beta1", webhook_fn=fn)
    # annotations are part of the conversion contract — preserved verbatim
    assert out[0]["metadata"]["annotations"] == annotations


def test_discover_conversion_service_parses_crd_clientconfig():
    """Discover conversion service parses crd clientconfig."""
    crd_list = {
        "items": [
            {"spec": {"group": "other.io", "names": {"kind": "DynamoGraphDeployment"}}},
            {
                "spec": {
                    "group": "nvidia.com",
                    "names": {"kind": "DynamoGraphDeployment"},
                    "conversion": {
                        "webhook": {
                            "clientConfig": {
                                "service": {
                                    "name": "dynamo-operator-webhook",
                                    "namespace": "dynamo-system",
                                    "port": 443,
                                    "path": "/convert",
                                }
                            }
                        }
                    },
                }
            },
        ]
    }
    svc = c.discover_conversion_service("DynamoGraphDeployment", crd_list=crd_list)
    assert svc == {
        "name": "dynamo-operator-webhook",
        "namespace": "dynamo-system",
        "port": 443,
        "path": "/convert",
    }


def test_discover_conversion_service_errors_when_no_webhook():
    """Discover conversion service errors when no webhook."""
    crd_list = {
        "items": [{"spec": {"group": "nvidia.com", "names": {"kind": "DynamoModel"}}}]
    }
    with pytest.raises(c.ConversionError, match="DynamoModel"):
        c.discover_conversion_service("DynamoModel", crd_list=crd_list)


def test_main_reads_input_converts_and_writes_output(tmp_path, monkeypatch):
    """Main reads input converts and writes output."""
    in_path = tmp_path / "in.yaml"
    out_path = tmp_path / "out.yaml"
    in_path.write_text(
        "apiVersion: nvidia.com/v1alpha1\n"
        "kind: DynamoGraphDeployment\n"
        "metadata:\n  name: a\n"
        "spec:\n  services: {}\n"
    )
    beta = {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "a"},
        "spec": {"components": []},
    }
    fn = _echo_webhook_fn_factory({"DynamoGraphDeployment": [beta]})
    monkeypatch.setattr(c, "default_webhook_fn", fn)

    rc = c.main(["-i", str(in_path), "-o", str(out_path)])
    assert rc == 0

    written = c.load_docs(out_path.read_text())
    assert written[0]["apiVersion"] == "nvidia.com/v1beta1"
    assert written[0]["spec"] == {"components": []}


def test_call_webhook_builds_proxy_path_and_posts(monkeypatch):
    """Call webhook builds proxy path and posts."""
    captured = {}

    def fake_kubectl(args):
        """Stub _kubectl that records the invocation and returns a canned response."""
        captured["args"] = args
        return "{}"

    monkeypatch.setattr(c, "_kubectl", fake_kubectl)
    service = {"name": "wh", "namespace": "ns", "port": 443, "path": "/convert"}
    out = c.call_webhook({"x": 1}, service)
    assert out == {}
    args = captured["args"]
    assert args[0] == "create"
    assert args[1] == "--raw"
    assert args[2] == "/api/v1/namespaces/ns/services/https:wh:443/proxy/convert"
    assert args[3] == "-f"


def test_normalize_drops_nulls_and_empties_recursively():
    """Normalize drops nulls and empties recursively."""
    obj = {"a": 1, "b": None, "c": {}, "d": [], "e": {"f": None, "g": 2}}
    assert c_test_normalize(obj) == {"a": 1, "e": {"g": 2}}


REPO_ROOT = _Path(__file__).resolve().parents[3]

CONFORMANCE_NAMES = ["agg", "disagg", "disagg_router", "disagg_planner"]


def _cluster_available() -> bool:
    """Return True if a Kubernetes cluster is reachable (gates conformance tests)."""
    if shutil.which("kubectl") is None:
        return False
    try:
        _sp.run(
            ["kubectl", "get", "crd", "dynamographdeployments.nvidia.com"],
            capture_output=True,
            check=True,
            timeout=5,
        )
        return True
    except (_sp.CalledProcessError, _sp.TimeoutExpired):
        return False


requires_cluster = pytest.mark.skipif(
    not _cluster_available(),
    reason="requires a cluster with the Dynamo conversion webhook installed",
)


@requires_cluster
@pytest.mark.timeout(60)
@pytest.mark.parametrize("name", CONFORMANCE_NAMES)
def test_conformance_vllm_example_pair(name):
    """Conformance vllm example pair."""
    alpha_path = REPO_ROOT / f"examples/backends/vllm/deploy/{name}.yaml"
    beta_path = REPO_ROOT / f"examples/backends/vllm/deploy/v1beta1/{name}.yaml"
    if not alpha_path.exists() or not beta_path.exists():
        pytest.skip(f"missing example pair for {name}")

    converted = c.convert_docs(
        c.load_docs(alpha_path.read_text()), target="nvidia.com/v1beta1"
    )
    expected = c.load_docs(beta_path.read_text())

    assert len(converted) == len(expected)
    for got, want in zip(converted, expected, strict=True):
        got_n = c_test_normalize(got)
        want_n = c_test_normalize(want)
        assert got_n.get("apiVersion") == want_n.get("apiVersion")
        assert got_n.get("kind") == want_n.get("kind")
        assert got_n.get("metadata", {}).get("name") == want_n.get("metadata", {}).get(
            "name"
        )
        assert got_n.get("spec") == want_n.get("spec"), (
            f"spec mismatch for {name}; webhook output diverged from the "
            f"committed v1beta1 fixture (tool bug or example drift)"
        )


def test_convert_docs_raises_when_webhook_returns_wrong_count():
    """Convert docs raises when webhook returns wrong count."""
    alpha = {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "a"},
        "spec": {},
    }
    fn = _echo_webhook_fn_factory(
        {"DynamoGraphDeployment": []}
    )  # 0 returned for 1 input
    with pytest.raises(c.ConversionError, match="expected 1"):
        c.convert_docs([alpha], target="nvidia.com/v1beta1", webhook_fn=fn)


def test_convert_docs_multi_kind_interleaved_preserves_order():
    """Convert docs multi kind interleaved preserves order."""
    dgd = {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "g"},
        "spec": {},
    }
    cm = {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "cm"}}
    dcd = {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoComponentDeployment",
        "metadata": {"name": "d"},
        "spec": {},
    }
    beta_dgd = {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "g"},
        "spec": {"components": []},
    }
    beta_dcd = {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoComponentDeployment",
        "metadata": {"name": "d"},
        "spec": {},
    }
    fn = _echo_webhook_fn_factory(
        {"DynamoGraphDeployment": [beta_dgd], "DynamoComponentDeployment": [beta_dcd]}
    )
    out = c.convert_docs([dgd, cm, dcd], target="nvidia.com/v1beta1", webhook_fn=fn)
    assert [o["kind"] for o in out] == [
        "DynamoGraphDeployment",
        "ConfigMap",
        "DynamoComponentDeployment",
    ]
    assert [o["metadata"]["name"] for o in out] == ["g", "cm", "d"]
    assert out[0]["apiVersion"] == "nvidia.com/v1beta1"
    assert out[1] == cm  # passthrough unchanged
    assert out[2]["apiVersion"] == "nvidia.com/v1beta1"


def test_dump_docs_quotes_yaml11_ambiguous_strings():
    # go-yaml (kubectl/apiserver, YAML 1.1) reads y/n/Y/N as booleans; PyYAML
    # does not, so unquoted output is re-parsed as bool and breaks the
    # "EnvVar.value must be string" contract on apply. dump_docs must quote them.
    """Dump docs quotes yaml11 ambiguous strings."""
    docs = [
        {
            "apiVersion": "nvidia.com/v1beta1",
            "kind": "DynamoGraphDeployment",
            "spec": {
                "env": [
                    {"name": "A", "value": "y"},
                    {"name": "B", "value": "n"},
                    {"name": "C", "value": "Y"},
                    {"name": "D", "value": "true"},
                    {"name": "E", "value": "off"},
                ]
            },
        }
    ]
    out = c.dump_docs(docs)
    assert "value: 'y'" in out
    assert "value: 'n'" in out
    assert "value: 'Y'" in out
    assert "value: 'true'" in out
    assert "value: 'off'" in out
    assert "value: y\n" not in out
    assert "value: n\n" not in out
