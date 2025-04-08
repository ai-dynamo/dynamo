import os

from dynamo.sdk.lib.config import ServiceConfig


def test_service_config_subscribed_correctly():
    ServiceConfig._instance = None
    os.environ[
        "DYNAMO_SERVICE_CONFIG"
    ] = """
{"Common": {"model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "block-size": 64, "max-model-len": 16384}, "Frontend": {"served_model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "endpoint": "dynamo.Processor.chat/completions", "port": 8000}, "Processor": {"router": "round-robin", "common-configs": ["model", "block-size", "max-model-len"]}, "VllmWorker": {"enforce-eager": true, "max-num-batched-tokens": 16384, "enable-prefix-caching":
true, "router": "random", "tensor-parallel-size": 1, "ServiceArgs": {"workers": 1}, "common-configs": ["model", "block-size", "max-model-len"]}}
"""
    service_config = ServiceConfig.get_instance()
    vllm_worker_args = service_config.as_args("VllmWorker")
    assert any(
        arg == f"--{key}"
        for key in ["model", "block-size", "max-model-len"]
        for arg in vllm_worker_args
    )


def test_service_config_not_subscribed():
    ServiceConfig._instance = None
    os.environ[
        "DYNAMO_SERVICE_CONFIG"
    ] = """
{"Common": {"model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "block-size": 64, "max-model-len": 16384}, "Frontend": {"served_model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "endpoint": "dynamo.Processor.chat/completions", "port": 8000}, "Processor": {"router": "round-robin", "common-configs": ["model", "block-size", "max-model-len"]}, "VllmWorker": {"enforce-eager": true, "max-num-batched-tokens": 16384, "enable-prefix-caching":
true, "router": "random", "tensor-parallel-size": 1, "ServiceArgs": {"workers": 1}}}
"""
    service_config = ServiceConfig.get_instance()
    vllm_worker_args = service_config.as_args("VllmWorker")
    assert all(
        not arg == f"--{key}"
        for key in ["model", "block-size", "max-model-len"]
        for arg in vllm_worker_args
    )
