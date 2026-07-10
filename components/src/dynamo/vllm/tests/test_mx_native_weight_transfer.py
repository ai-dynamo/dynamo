import pytest

from dynamo.vllm.handlers import DecodeWorkerHandler


class _Engine:
    def __init__(self):
        self.calls = []

    async def start_weight_update(self, *, is_checkpoint_format):
        self.calls.append(("start", is_checkpoint_format))

    async def update_weights(self, request):
        self.calls.append(("update", request.update_info))

    async def finish_weight_update(self):
        self.calls.append(("finish",))

    async def collective_rpc(self, *args, **kwargs):
        raise AssertionError("legacy collective_rpc path should not be used")


@pytest.mark.asyncio
async def test_native_mx_route_drives_vllm_weight_transfer(monkeypatch):
    monkeypatch.setenv("DYN_MX_NATIVE_WEIGHT_TRANSFER", "1")
    engine = _Engine()
    handler = object.__new__(DecodeWorkerHandler)
    handler.engine_client = engine

    responses = [
        response
        async for response in handler.update_weights_via_mx(
            {
                "version": 7,
                "mx_config": {
                    "timeout_seconds": 45,
                    "moe_expert_filter": True,
                    "ep_world_size": 2,
                    "ep_rank": 1,
                    "num_experts": 128,
                },
            }
        )
    ]

    assert responses == [
        {
            "status": "ok",
            "version": 7,
            "workers_ok": 1,
            "workers_total": 1,
        }
    ]
    assert engine.calls[0] == ("start", True)
    assert engine.calls[1][0] == "update"
    assert engine.calls[1][1] == {
        "version": 7,
        "min_version": 7,
        "timeout_seconds": 45.0,
        "moe_expert_filter": True,
        "expert_placement": "linear",
        "ep_world_size": 2,
        "ep_rank": 1,
        "num_experts": 128,
    }
    assert engine.calls[2] == ("finish",)


@pytest.mark.asyncio
async def test_native_mx_route_finishes_after_update_failure(monkeypatch):
    monkeypatch.setenv("DYN_MX_NATIVE_WEIGHT_TRANSFER", "1")

    class _FailingEngine(_Engine):
        async def update_weights(self, request):
            self.calls.append(("update", request.update_info))
            raise RuntimeError("receive failed")

    engine = _FailingEngine()
    handler = object.__new__(DecodeWorkerHandler)
    handler.engine_client = engine

    responses = [
        response
        async for response in handler.update_weights_via_mx(
            {"version": 9, "mx_config": {}}
        )
    ]

    assert responses[0]["status"] == "error"
    assert responses[0]["version"] == 9
    assert engine.calls[-1] == ("finish",)
