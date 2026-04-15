import pytest
from dynamo.trtllm.health_check import TrtllmHealthCheckPayload


def test_trtllm_health_check_payload_has_no_disagg_params():
    """Standard TrtllmHealthCheckPayload should NOT include disaggregated_params."""
    payload = TrtllmHealthCheckPayload().to_dict()
    assert "disaggregated_params" not in payload
    assert "prefill_result" not in payload
    assert "token_ids" in payload
