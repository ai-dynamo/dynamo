# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging

import pytest
import torch

from dynamo.experimental.workflow import (
    DeploymentSpec,
    InlineBinding,
    NixlLeaseRegistry,
    NixlTensorCarrier,
    NixlTensorFanout,
    NixlTensorRef,
    RemoteBinding,
    StageContract,
    Workflow,
    WorkflowExecutionError,
    WorkflowOrchestrator,
    compile_workflow,
)
from dynamo.experimental.workflow.dispatcher import StageDispatcher
from dynamo.experimental.workflow.nixl import EmbeddingTransferRef
from dynamo.experimental.workflow.perf import WorkflowPerfTracer
from dynamo.experimental.workflow.remote import NixlCarriedValue

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.core,
]


class _Metadata:
    def __init__(self, value):
        self.value = value

    def model_dump(self):
        return dict(self.value)


class _Descriptor:
    def __init__(self, tensor):
        self.tensor = tensor
        self.connection = None

    @property
    def is_registered(self):
        return self.connection is not None

    def register_with_connector(self, connection):
        self.connection = connection

    def deregister_with_connector(self, connection):
        assert self.connection is connection
        self.connection = None


class _Readable:
    def __init__(self, key, descriptor):
        self.key = key
        self.descriptor = descriptor
        self.completed = asyncio.Event()
        self.released = False

    def metadata(self):
        return _Metadata(
            {
                "key": self.key,
                "descriptors": [
                    {
                        "device": "cpu",
                        "ptr": 1,
                        "size": self.descriptor.tensor.numel()
                        * self.descriptor.tensor.element_size(),
                    }
                ],
            }
        )

    async def wait_for_completion(self):
        await self.completed.wait()

    def __exit__(self, exc_type, exc_value, traceback):
        self.released = True


class _Read:
    def __init__(self, source, destination):
        self.source = source
        self.destination = destination
        self.released = False

    async def wait_for_completion(self):
        destination = self.destination.tensor.view(torch.uint8).flatten()
        source = self.source.descriptor.tensor.view(torch.uint8).flatten()
        destination.copy_(source[: destination.numel()])
        self.source.completed.set()

    def __exit__(self, exc_type, exc_value, traceback):
        self.released = True


class _RdmaMetadata:
    @classmethod
    def model_validate(cls, value):
        return dict(value)


class _NixlModule:
    Descriptor = _Descriptor
    RdmaMetadata = _RdmaMetadata


class _Connector:
    def __init__(self):
        self.readables = {}
        self.reads = []

    async def create_readable(self, descriptor):
        key = f"transfer-{len(self.readables)}"
        readable = _Readable(key, descriptor)
        self.readables[key] = readable
        return readable

    async def _create_connection(self):
        return self

    async def begin_read(self, metadata, descriptor):
        read = _Read(self.readables[metadata["key"]], descriptor)
        self.reads.append(read)
        return read


class _IsolatedNixlModule(_NixlModule):
    connectors = []
    connector_options = []

    @classmethod
    def Connector(cls, **options):
        connector = _Connector()
        cls.connectors.append(connector)
        cls.connector_options.append(options)
        return connector


async def _wait_for_no_leases(carrier):
    for _ in range(20):
        if carrier.active_leases == 0:
            return
        await asyncio.sleep(0)
    raise AssertionError("NIXL lease did not complete")


async def _wait_for_registry_empty(registry: NixlLeaseRegistry) -> None:
    for _ in range(20):
        if registry.active_count == 0:
            return
        await asyncio.sleep(0)
    raise AssertionError("NIXL registry did not release completed reads")


async def test_tensor_carrier_round_trip_keeps_per_transfer_lease() -> None:
    connector = _Connector()
    carrier = NixlTensorCarrier(
        connector=connector,
        nixl_module=_NixlModule,
        torch_module=torch,
    )
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    reference = await carrier.export_tensor(source, "classifier.embedding")
    assert carrier.active_leases == 1
    received = await carrier.import_tensor(reference)
    await _wait_for_no_leases(carrier)

    assert torch.equal(received, source)
    assert connector.readables["transfer-0"].released
    assert connector.reads[0].released


async def test_one_logical_tensor_can_have_independent_consumer_leases() -> None:
    connector = _Connector()
    carrier = NixlTensorCarrier(
        connector=connector,
        nixl_module=_NixlModule,
        torch_module=torch,
    )
    source = torch.ones((2, 8), dtype=torch.float16)

    references = await carrier.export_tensor_fanout(
        source, ("classifier.embedding", "generator.embedding")
    )
    classifier = NixlTensorRef.from_dict(references["classifier.embedding"])
    generator = NixlTensorRef.from_dict(references["generator.embedding"])
    fanout = NixlTensorFanout(
        {
            classifier.transfer_id: classifier,
            generator.transfer_id: generator,
        }
    )

    assert carrier.active_leases == 2
    assert NixlTensorFanout.from_dict(fanout.to_dict()) == fanout
    assert (
        fanout.for_transfer("classifier.embedding").lease_id
        != fanout.for_transfer("generator.embedding").lease_id
    )

    await carrier.import_tensor(classifier.to_dict())
    assert carrier.active_leases == 2
    assert not connector.readables["transfer-0"].released
    assert not connector.readables["transfer-1"].released
    await carrier.import_tensor(generator.to_dict())
    await _wait_for_no_leases(carrier)
    assert connector.readables["transfer-0"].released
    assert connector.readables["transfer-1"].released


async def test_default_carrier_isolates_each_exported_edge() -> None:
    _IsolatedNixlModule.connectors = []
    _IsolatedNixlModule.connector_options = []
    carrier = NixlTensorCarrier(
        nixl_module=_IsolatedNixlModule,
        torch_module=torch,
    )

    await carrier.export_tensor_fanout(
        torch.ones((2, 8), dtype=torch.float16),
        ("classifier.embedding", "generator.embedding"),
    )

    # One long-lived connector receives tensors; each exported edge gets an
    # immutable transfer agent so concurrent registrations cannot race.
    assert len(_IsolatedNixlModule.connectors) == 3
    assert _IsolatedNixlModule.connector_options == [
        {"enable_progress_thread": False},
        {"enable_progress_thread": False},
        {"enable_progress_thread": False},
    ]
    assert not _IsolatedNixlModule.connectors[0].readables
    assert len(_IsolatedNixlModule.connectors[1].readables) == 1
    assert len(_IsolatedNixlModule.connectors[2].readables) == 1
    for connector in _IsolatedNixlModule.connectors[1:]:
        next(iter(connector.readables.values())).completed.set()
    await _wait_for_no_leases(carrier)


def test_progress_thread_can_be_enabled_for_every_carrier_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _IsolatedNixlModule.connectors = []
    _IsolatedNixlModule.connector_options = []
    monkeypatch.setenv("DYN_NIXL_PROGRESS_THREAD", "1")

    NixlTensorCarrier(
        nixl_module=_IsolatedNixlModule,
        torch_module=torch,
    )

    assert _IsolatedNixlModule.connector_options == [{"enable_progress_thread": True}]


async def test_send_pool_reuses_slot_after_all_fanout_reads() -> None:
    connector = _Connector()
    carrier = NixlTensorCarrier(
        connector=connector,
        nixl_module=_NixlModule,
        torch_module=torch,
        send_pool_capacity=1,
        send_pool_bytes=64,
    )
    source = torch.ones((2, 8), dtype=torch.float16)

    references = await carrier.export_tensor_fanout(
        source, ("classifier.embedding", "generator.embedding")
    )
    assert all(
        reference["rdma_metadata"]["descriptors"][0]["size"] == 32
        for reference in references.values()
    )
    blocked = asyncio.create_task(carrier.export_tensor(source, "next.embedding"))
    await asyncio.sleep(0)
    assert not blocked.done()

    first_readables = list(connector.readables.values())
    first_readables[0].completed.set()
    await asyncio.sleep(0)
    assert not blocked.done()
    first_readables[1].completed.set()
    await blocked
    next_readable = list(connector.readables.values())[-1]
    next_readable.completed.set()
    await _wait_for_no_leases(carrier)
    await carrier.close()


async def test_send_pool_trace_correlates_export_import_and_release(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        "dynamo.experimental.workflow.nixl.WORKFLOW_PERF_TRACE",
        WorkflowPerfTracer(enabled=True, sample_every=1),
    )
    connector = _Connector()
    carrier = NixlTensorCarrier(
        connector=connector,
        nixl_module=_NixlModule,
        torch_module=torch,
        send_pool_capacity=1,
        send_pool_bytes=64,
    )

    with caplog.at_level(logging.INFO):
        reference = await carrier.export_tensor(
            torch.ones((2, 8), dtype=torch.float16), "decoder.embedding"
        )
        await carrier.import_tensor(reference)
        await _wait_for_no_leases(carrier)

    payloads = [
        json.loads(record.getMessage().removeprefix("workflow_perf "))
        for record in caplog.records
        if record.getMessage().startswith("workflow_perf ")
    ]
    lease_id = reference["lease_id"]
    lease_events = {
        payload["event"] for payload in payloads if payload["trace_id"] == lease_id
    }
    assert lease_events == {
        "nixl.export",
        "nixl.import",
        "nixl.lease_release",
        "nixl.producer_wait",
    }


async def test_lease_registry_retains_unread_operation_after_timeout() -> None:
    registry = NixlLeaseRegistry(timeout_s=0.01)
    operation = _Readable("never-read", _Descriptor(torch.ones(1)))

    registry.track("lease-1", operation, torch.ones(1024))
    assert registry.active_count == 1
    await asyncio.sleep(0.02)

    assert registry.active_count == 1
    assert not operation.released

    operation.completed.set()
    await _wait_for_registry_empty(registry)

    assert operation.released


async def test_lease_registry_close_does_not_release_uncertain_read() -> None:
    registry = NixlLeaseRegistry(timeout_s=1.0)
    operation = _Readable("active-read", _Descriptor(torch.ones(1)))

    registry.track("lease-1", operation, torch.ones(1024))
    await registry.close()

    assert registry.active_count == 1
    assert not operation.released

    operation.completed.set()
    await _wait_for_registry_empty(registry)
    assert operation.released


def test_tensor_reference_rejects_unknown_wire_fields() -> None:
    reference = NixlTensorRef(
        transfer_id="stage.input",
        lease_id="lease",
        shape=(2, 3),
        dtype="float32",
        device="cpu",
        rdma_metadata={"opaque": True},
    ).to_dict()
    reference["fallback"] = "inline"

    with pytest.raises(WorkflowExecutionError, match="unknown fields"):
        NixlTensorRef.from_dict(reference)


def test_embedding_transfer_reference_uses_existing_request_shape() -> None:
    reference = EmbeddingTransferRef(
        shape=(3, 8),
        dtype="torch.bfloat16",
        serialized_request="opaque-write-handshake",
        transfer_id="generator.embedding",
    )

    assert reference.to_dict() == {
        "embeddings_shape": [3, 8],
        "embedding_dtype_str": "bfloat16",
        "serialized_request": "opaque-write-handshake",
    }
    assert (
        EmbeddingTransferRef.from_dict(
            reference.to_dict(), transfer_id="generator.embedding"
        )
        == reference
    )


ENCODER = StageContract(
    id="encoder",
    inputs={"request"},
    outputs={"embedding"},
)
CLASSIFIER = StageContract(
    id="classifier",
    inputs={"embedding"},
    outputs={"scores"},
)
GENERATOR = StageContract(
    id="generator",
    inputs={"embedding"},
    outputs={"text"},
)


def _tensor_workflow() -> Workflow:
    workflow = Workflow("nixl-fanout")
    request = workflow.input("request")
    encoder = workflow.stage("encoder", ENCODER, request=request)
    classifier = workflow.stage("classifier", CLASSIFIER, embedding=encoder.embedding)
    generator = workflow.stage("generator", GENERATOR, embedding=encoder.embedding)
    workflow.output("scores", classifier.scores)
    workflow.output("text", generator.text)
    return workflow


class _RemoteTensorInvoker:
    def __init__(self, role: str) -> None:
        self.role = role
        self.calls: list[tuple[dict, dict]] = []

    async def run(self, stage_id, contract, inputs, context, output_transfers):
        self.calls.append((dict(inputs), dict(output_transfers)))
        assert stage_id == self.role
        if self.role == "encoder":
            transfer_ids = output_transfers.get("embedding", ("classifier.embedding",))
            return {
                "embedding": NixlCarriedValue(
                    NixlTensorFanout(
                        {
                            transfer_id: NixlTensorRef(
                                transfer_id=transfer_id,
                                lease_id=f"lease-{transfer_id}",
                                shape=(4, 8),
                                dtype="float32",
                                device="cuda:0",
                                rdma_metadata={"opaque": transfer_id},
                            )
                            for transfer_id in transfer_ids
                        }
                    ).to_dict()
                )
            }
        assert isinstance(inputs["embedding"], NixlCarriedValue)
        reference = NixlTensorRef.from_dict(inputs["embedding"].value)
        assert reference.transfer_id == f"{self.role}.embedding"
        if self.role == "classifier":
            return {"scores": {"ok": 1.0}}
        return {"text": "generated"}


async def test_graph_dispatches_one_remote_nixl_reference_per_consumer() -> None:
    endpoints = {
        stage_id: f"workflows.{stage_id}.generate"
        for stage_id in ("encoder", "classifier", "generator")
    }
    plan = compile_workflow(
        _tensor_workflow(),
        DeploymentSpec.remote(tensor_carrier="nixl", **endpoints),
    )
    invokers = {
        endpoints[stage_id]: _RemoteTensorInvoker(stage_id) for stage_id in endpoints
    }
    dispatcher = StageDispatcher(plan, {}, invokers)
    executor = WorkflowOrchestrator(plan, dispatcher)

    assert await executor.run({"request": {"token_ids": [1]}}) == {
        "scores": {"ok": 1.0},
        "text": "generated",
    }
    assert invokers[endpoints["encoder"]].calls[0][1] == {
        "embedding": ("classifier.embedding", "generator.embedding")
    }


async def test_carried_tensor_cannot_cross_inline_boundary() -> None:
    workflow = Workflow("mixed-nixl")
    request = workflow.input("request")
    encoder = workflow.stage("encoder", ENCODER, request=request)
    classifier = workflow.stage("classifier", CLASSIFIER, embedding=encoder.embedding)
    workflow.output("scores", classifier.scores)
    plan = compile_workflow(
        workflow,
        DeploymentSpec(
            {
                "encoder": RemoteBinding(
                    "workflows.encoder.generate", tensor_carrier="nixl"
                ),
                "classifier": InlineBinding("classifier"),
            }
        ),
    )

    class InlineClassifier:
        contract = CLASSIFIER

        async def run(self, inputs, context):
            raise AssertionError("inline runner must not receive a NIXL reference")

    dispatcher = StageDispatcher(
        plan,
        {"classifier": InlineClassifier()},
        {"workflows.encoder.generate": _RemoteTensorInvoker("encoder")},
    )
    executor = WorkflowOrchestrator(plan, dispatcher)

    with pytest.raises(WorkflowExecutionError, match="must target a NIXL remote"):
        await executor.run({"request": {"token_ids": [1]}})


async def test_carried_tensor_cannot_be_a_workflow_output() -> None:
    workflow = Workflow("nixl-output")
    request = workflow.input("request")
    encoder = workflow.stage("encoder", ENCODER, request=request)
    workflow.output("embedding", encoder.embedding)
    endpoint = "workflows.encoder.generate"
    plan = compile_workflow(
        workflow,
        DeploymentSpec.remote(
            tensor_carrier="nixl",
            encoder=endpoint,
        ),
    )
    dispatcher = StageDispatcher(
        plan,
        {},
        {endpoint: _RemoteTensorInvoker("encoder")},
    )
    executor = WorkflowOrchestrator(plan, dispatcher)

    with pytest.raises(WorkflowExecutionError, match="cannot expose a NIXL tensor"):
        await executor.run({"request": {"token_ids": [1]}})
