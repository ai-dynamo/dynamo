from copy import deepcopy
from typing import Optional
import pytest

from benchmarks.profiler.utils.config import (
    Config, 
    SGLangConfigModifier, 
    TrtllmConfigModifier, 
    VllmV1ConfigModifier, 
    Service,
    get_service_name_from_sub_component_type, 
    break_arguments,  
    parse_override_engine_args
)
from benchmarks.profiler.utils.defaults import (
    DEFAULT_MODEL_NAME,
)

@pytest.fixture
def vllm_config():
    return {
        "metadata": {
            "name": "vllm-disagg"
        },
        "spec": {
            "services": {
                "Frontend": {
                    "dynamoNamespace": "vllm-disagg",
                    "componentType": "frontend",
                    "replicas": 1,
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.4.1"
                        }
                    }
                },
                "Planner": {
                    "dynamoNamespace": "vllm-disagg",
                    "componentType": "planner",
                },
                "PrefillWorker": {
                    "dynamoNamespace": "vllm-disagg",
                    "envFromSecret": "hf-token-secret",
                    "componentType": "worker",
                    "subComponentType": "prefill",
                    "replicas": 2,
                    "resources": {
                        "limits": {
                            "gpu": "1"
                        }
                    },
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.4.1",
                            "command": ["python3"],
                            "args": ["-m", "dynamo.vllm", "--model", "openai/gpt-oss-120b", "--is-prefill-worker", "--enable-prefix-caching"]
                        }
                    }
                },
                "DecodeWorker": {
                    "dynamoNamespace": "vllm-disagg",
                    "envFromSecret": "hf-token-secret",
                    "componentType": "worker",
                    "subComponentType": "decode",
                    "replicas": 2,
                    "resources": {
                        "limits": {
                            "gpu": "1"
                        }
                    },
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.4.1",
                            "command": ["python3"],
                            "args": ["-m", "dynamo.vllm", "--model", "openai/gpt-oss-120b"]
                        }
                    }
                },
            }
        }
    }

@pytest.fixture
def vllm_config_no_sub_component_type(vllm_config: dict):
    new_vllm_config = deepcopy(vllm_config)
    del new_vllm_config["spec"]["services"]["PrefillWorker"]["subComponentType"]
    del new_vllm_config["spec"]["services"]["DecodeWorker"]["subComponentType"]
    new_vllm_config["spec"]["services"]["VllmPrefillWorker"] = new_vllm_config["spec"]["services"]["PrefillWorker"]
    del new_vllm_config["spec"]["services"]["PrefillWorker"]
    new_vllm_config["spec"]["services"]["VllmDecodeWorker"] = new_vllm_config["spec"]["services"]["DecodeWorker"]
    del new_vllm_config["spec"]["services"]["DecodeWorker"]
    return new_vllm_config

@pytest.fixture
def sglang_config():
    return {
        "metadata": {
            "name": "sglang-disagg"
        },
        "spec": {
            "services": {
                "Frontend": {
                    "dynamoNamespace": "sglang-disagg",
                    "componentType": "frontend",
                    "replicas": 1,
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": "nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.4.1"
                        }
                    }
                },
                "Planner": {
                    "dynamoNamespace": "sglang-disagg",
                    "componentType": "planner",
                },
                "PrefillWorker": {
                    "dynamoNamespace": "sglang-disagg",
                    "envFromSecret": "hf-token-secret",
                    "componentType": "worker",
                    "subComponentType": "prefill",
                    "replicas": 2,
                    "resources": {
                        "limits": {
                            "gpu": "1"
                        }
                    },
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": "nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.4.1",
                            "command": ["python3"],
                            "args": ["-m", "dynamo.sglang", "--model-path", "openai/gpt-oss-120b", "--served-model-name", "openai/gpt-oss-120b", "--page-size", "16", "--tp", "1", "--trust-remote-code", "--skip-tokenizer-init", "--disaggregation-mode", "prefill", "--disaggregation-transfer-backend", "nixl"]
                        }
                    }
                },
                "DecodeWorker": {
                    "dynamoNamespace": "sglang-disagg",
                    "envFromSecret": "hf-token-secret",
                    "componentType": "worker",
                    "subComponentType": "decode",
                    "replicas": 2,
                    "resources": {
                        "limits": {
                            "gpu": "1"
                        }
                    },
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": "nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.4.1",
                            "command": ["python3"],
                            "args": ["-m", "dynamo.sglang", "--model-path", "openai/gpt-oss-120b", "--served-model-name", "openai/gpt-oss-120b", "--page-size", "16", "--tp", "1", "--trust-remote-code", "--skip-tokenizer-init", "--disaggregation-mode", "decode", "--disaggregation-transfer-backend", "nixl"]
                        }
                    }
                }
            },
        },
    }

@pytest.fixture
def sglang_config_no_sub_component_type(sglang_config: dict):
    new_sglang_config = deepcopy(sglang_config)
    del new_sglang_config["spec"]["services"]["PrefillWorker"]["subComponentType"]
    del new_sglang_config["spec"]["services"]["DecodeWorker"]["subComponentType"]
    new_sglang_config["spec"]["services"]["SGLangPrefillWorker"] = new_sglang_config["spec"]["services"]["PrefillWorker"]
    del new_sglang_config["spec"]["services"]["PrefillWorker"]
    new_sglang_config["spec"]["services"]["SGLangDecodeWorker"] = new_sglang_config["spec"]["services"]["DecodeWorker"]
    del new_sglang_config["spec"]["services"]["DecodeWorker"]
    return new_sglang_config

@pytest.fixture
def trtllm_config():
        return {
        "metadata": {
            "name": "trtllm-disagg"
        },
        "spec": {
            "services": {
                "Frontend": {
                    "dynamoNamespace": "trtllm-disagg",
                    "componentType": "frontend",
                    "replicas": 2,
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": "nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.4.1"
                        }
                    }
                },
                "Planner": {
                    "dynamoNamespace": "trtllm-disagg",
                    "componentType": "planner",
                },
                "DecodeWorker": {
                    "dynamoNamespace": "trtllm-disagg",
                    "envFromSecret": "hf-token-secret",
                    "componentType": "worker",
                    "subComponentType": "decode",
                    "replicas": 2,
                    "resources": {
                        "limits": {
                            "gpu": "1"
                        }
                    },
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": "nvcr.io/nvidia/ai-dynamo/trtllm-runtime:0.4.1",
                            "command": ["python3"],
                            "args": ["-m", "dynamo.trtllm", "--model-path", "openai/gpt-oss-120b", "--served-model-name", "openai/gpt-oss-120b", "--extra-engine-args", "engine_configs/decode.yaml", "--disaggregation-mode", "decode", "--disaggregation-strategy", "decode_first"],
                        }
                    }
                },
                "PrefillWorker": {
                    "dynamoNamespace": "trtllm-disagg",
                    "envFromSecret": "hf-token-secret",
                    "componentType": "worker",
                    "subComponentType": "prefill",
                    "replicas": 1,
                    "resources": {
                        "limits": {
                            "gpu": "1"
                        }
                    },
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": "nvcr.io/nvidia/ai-dynamo/trtllm-runtime:0.4.1",
                            "command": ["python3"],
                            "args": ["-m", "dynamo.trtllm", "--model-path", "openai/gpt-oss-120b", "--served-model-name", "openai/gpt-oss-120b", "--extra-engine-args", "engine_configs/prefill.yaml", "--disaggregation-mode", "prefill", "--disaggregation-strategy", "decode_first"],
                        }
                    }
                }
            },
        },
    }

@pytest.fixture
def trtllm_config_no_sub_component_type(trtllm_config: dict):
    new_trtllm_config = deepcopy(trtllm_config)
    del new_trtllm_config["spec"]["services"]["PrefillWorker"]["subComponentType"]
    del new_trtllm_config["spec"]["services"]["DecodeWorker"]["subComponentType"]
    new_trtllm_config["spec"]["services"]["TRTLLMPrefillWorker"] = new_trtllm_config["spec"]["services"]["PrefillWorker"]
    del new_trtllm_config["spec"]["services"]["PrefillWorker"]
    new_trtllm_config["spec"]["services"]["TRTLLMDecodeWorker"] = new_trtllm_config["spec"]["services"]["DecodeWorker"]
    del new_trtllm_config["spec"]["services"]["DecodeWorker"]
    return new_trtllm_config

def test_vllm_config_get_model_name(vllm_config, vllm_config_no_sub_component_type):
    config_modifier = VllmV1ConfigModifier()
    assert config_modifier.get_model_name(vllm_config) == "openai/gpt-oss-120b"

    # no extra pod spec returns default model name
    config_no_extra_pod_spec = {
        "metadata": {
            "name": "vllm-agg"
        },
        "spec": {
            "services": {
                "VllmDecodeWorker": {
                    "dynamoNamespace": "vllm-disagg",
                },
            }
        }
    }

    assert config_modifier.get_model_name(config_no_extra_pod_spec) == DEFAULT_MODEL_NAME

    config_no_model_arg = {
        "metadata": {
            "name": "vllm-agg"
        },
        "spec": {
            "services": {
                "VllmDecodeWorker": {
                    "dynamoNamespace": "vllm-disagg",
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.4.1",
                            "command": ["python3"],
                            "args": ["-m", "dynamo.vllm", "--model"]
                        }
                    }
                },
            }
        }
    }

    assert config_modifier.get_model_name(config_no_model_arg) == DEFAULT_MODEL_NAME

    assert config_modifier.get_model_name(vllm_config_no_sub_component_type) == "openai/gpt-oss-120b"


def test_sglang_config_get_model_name(sglang_config, sglang_config_no_sub_component_type):
    config_modifier = SGLangConfigModifier()
    assert config_modifier.get_model_name(sglang_config) == "openai/gpt-oss-120b"

    # no extra pod spec returns default model name
    config_no_extra_pod_spec = {
        "metadata": {
            "name": "sglang-agg"
        },
        "spec": {
            "services": {
                "SGLangDecodeWorker": {
                    "dynamoNamespace": "sglang-disagg",
                },
            }
        }
    }

    assert config_modifier.get_model_name(config_no_extra_pod_spec) == DEFAULT_MODEL_NAME

    config_no_model_arg = {
        "metadata": {
            "name": "sglang-agg"
        },
        "spec": {
            "services": {
                "SGLangDecodeWorker": {
                    "dynamoNamespace": "sglang-disagg",
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": "nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.4.1",
                            "command": ["python3"],
                            "args": ["-m", "dynamo.sglang", "--model-path", "openai/gpt-oss-120b"]
                        }
                    }
                },
            }
        }
    }

    assert config_modifier.get_model_name(config_no_model_arg) == DEFAULT_MODEL_NAME

    assert config_modifier.get_model_name(sglang_config_no_sub_component_type) == "openai/gpt-oss-120b"

def test_trtllm_config_get_model_name(trtllm_config, trtllm_config_no_sub_component_type):
    config_modifier = TrtllmConfigModifier()
    assert config_modifier.get_model_name(trtllm_config) == "openai/gpt-oss-120b"

        # no extra pod spec returns default model name
    config_no_extra_pod_spec = {
        "metadata": {
            "name": "trtllm-agg"
        },
        "spec": {
            "services": {
                "TRTLLMDecodeWorker": {
                    "dynamoNamespace": "trtllm-disagg",
                },
            }
        }
    }

    assert config_modifier.get_model_name(config_no_extra_pod_spec) == DEFAULT_MODEL_NAME

    config_no_model_arg = {
        "metadata": {
            "name": "trtllm-agg"
        },
        "spec": {
            "services": {
                "TRTLLMDecodeWorker": {
                    "dynamoNamespace": "trtllm-disagg",
                    "extraPodSpec": {
                        "mainContainer": {
                            "image": "nvcr.io/nvidia/ai-dynamo/trtllm-runtime:0.4.1",
                            "command": ["python3"],
                            "args": ["-m", "dynamo.trtllm", "--model-path", "openai/gpt-oss-120b"]
                        }
                    }
                },
            }
        }
    }

    assert config_modifier.get_model_name(config_no_model_arg) == DEFAULT_MODEL_NAME

    assert config_modifier.get_model_name(trtllm_config_no_sub_component_type) == "openai/gpt-oss-120b"

def test_vllm_config_convert_config_prefill(vllm_config, vllm_config_no_sub_component_type):
    config_modifier = VllmV1ConfigModifier()

    def get_prefill_config(config: dict) -> Config: 
        prefill_config = config_modifier.convert_config(config, "prefill")
        cfg = Config.model_validate(prefill_config)
        return cfg
    
    def validate_prefill_worker(prefill_worker: Service):
        assert prefill_worker.extraPodSpec is not None
        assert prefill_worker.extraPodSpec.mainContainer is not None

        args = break_arguments(prefill_worker.extraPodSpec.mainContainer.args)
        assert "--is-prefill-worker" not in args
        assert "--enable-prefix-caching" not in args
        assert "--no-enable-prefix-caching" in args

        assert prefill_worker.replicas == 1

    cfg = get_prefill_config(vllm_config)
    assert get_service_name_from_sub_component_type(cfg.spec.services, "decode") is None
    prefill_worker_name = get_service_name_from_sub_component_type(cfg.spec.services, "prefill")
    assert prefill_worker_name is not None
    prefill_worker = cfg.spec.services.get(prefill_worker_name)
    assert prefill_worker is not None
    validate_prefill_worker(prefill_worker)

    cfg = get_prefill_config(vllm_config_no_sub_component_type)
    prefill_worker = cfg.spec.services.get("VllmDecodeWorker")
    assert prefill_worker is not None
    assert cfg.spec.services.get("VllmPrefillWorker") is None
    validate_prefill_worker(prefill_worker)


def test_vllm_config_convert_config_decode(vllm_config, vllm_config_no_sub_component_type):
    config_modifier = VllmV1ConfigModifier()

    def get_decode_config(config: dict) -> Config:
        decode_config = config_modifier.convert_config(config, "decode")
        cfg = Config.model_validate(decode_config)
        return cfg


    def validate_decode_worker(decode_worker: Service):
        assert decode_worker.extraPodSpec is not None
        assert decode_worker.extraPodSpec.mainContainer is not None

        args = break_arguments(decode_worker.extraPodSpec.mainContainer.args)
        assert "--enable-prefix-caching" in args
        assert "--no-enable-prefix-caching" not in args

        assert decode_worker.replicas == 1

    cfg = get_decode_config(vllm_config)
    assert cfg.spec.services.get("Planner") is None
    decode_worker_name = get_service_name_from_sub_component_type(cfg.spec.services, "decode")
    assert decode_worker_name is not None
    assert get_service_name_from_sub_component_type(cfg.spec.services, "prefill") is None
    decode_worker = cfg.spec.services.get(decode_worker_name)
    assert decode_worker is not None
    validate_decode_worker(decode_worker)

    cfg = get_decode_config(vllm_config_no_sub_component_type)
    assert cfg.spec.services.get("Planner") is None
    decode_worker = cfg.spec.services.get("VllmDecodeWorker")
    assert decode_worker is not None
    assert cfg.spec.services.get("VllmPrefillWorker") is None
    validate_decode_worker(decode_worker)

def test_sglang_config_convert_config_prefill(sglang_config, sglang_config_no_sub_component_type):
    config_modifier = SGLangConfigModifier()

    def get_prefill_config(config: dict) -> Config: 
        prefill_config = config_modifier.convert_config(config, "prefill")
        cfg = Config.model_validate(prefill_config)
        return cfg
    
    def validate_prefill_worker(prefill_worker: Service):
        assert prefill_worker.extraPodSpec is not None
        assert prefill_worker.extraPodSpec.mainContainer is not None

        args = break_arguments(prefill_worker.extraPodSpec.mainContainer.args)
        assert "--disaggregation-mode" not in args
        assert "prefill" not in args
        assert "--disaggregation-transfer-backend" not in args
        assert "nixl" not in args
        assert "--disable-radix-cache" in args

        assert prefill_worker.replicas == 1

    cfg = get_prefill_config(sglang_config)
    assert cfg.spec.services.get("Planner") is None
    assert get_service_name_from_sub_component_type(cfg.spec.services, "decode") is None
    prefill_worker_name = get_service_name_from_sub_component_type(cfg.spec.services, "prefill")
    assert prefill_worker_name is not None
    prefill_worker = cfg.spec.services.get(prefill_worker_name)
    assert prefill_worker is not None
    validate_prefill_worker(prefill_worker)

    cfg = get_prefill_config(sglang_config_no_sub_component_type)
    assert cfg.spec.services.get("Planner") is None
    prefill_worker = cfg.spec.services.get("SGLangDecodeWorker")
    assert prefill_worker is not None
    assert cfg.spec.services.get("SGLangPrefillWorker") is None
    validate_prefill_worker(prefill_worker)

def test_sglang_config_convert_config_decode(sglang_config, sglang_config_no_sub_component_type):
    config_modifier = SGLangConfigModifier()

    def get_decode_config(config: dict) -> Config: 
        decode_config = config_modifier.convert_config(config, "decode")
        cfg = Config.model_validate(decode_config)
        return cfg
    
    def validate_decode_worker(decode_worker: Service):
        assert decode_worker.extraPodSpec is not None
        assert decode_worker.extraPodSpec.mainContainer is not None

        args = break_arguments(decode_worker.extraPodSpec.mainContainer.args)
        assert "--disaggregation-mode" not in args
        assert "prefill" not in args
        assert "--disaggregation-transfer-backend" not in args
        assert "nixl" not in args
        assert "--disable-radix-cache" not in args

        assert decode_worker.replicas == 1

    cfg = get_decode_config(sglang_config)
    assert cfg.spec.services.get("Planner") is None
    assert get_service_name_from_sub_component_type(cfg.spec.services, "prefill") is None
    decode_worker_name = get_service_name_from_sub_component_type(cfg.spec.services, "decode")
    assert decode_worker_name is not None
    decode_worker = cfg.spec.services.get(decode_worker_name)
    assert decode_worker is not None
    validate_decode_worker(decode_worker)

    cfg = get_decode_config(sglang_config_no_sub_component_type)
    assert cfg.spec.services.get("Planner") is None
    decode_worker = cfg.spec.services.get("SGLangDecodeWorker")
    assert decode_worker is not None
    assert cfg.spec.services.get("SGLangPrefillWorker") is None
    validate_decode_worker(decode_worker)

def test_trtllm_config_convert_config_prefill(trtllm_config, trtllm_config_no_sub_component_type):
    config_modifier = TrtllmConfigModifier()

    def get_prefill_config(config: dict) -> Config: 
        prefill_config = config_modifier.convert_config(config, "prefill")
        cfg = Config.model_validate(prefill_config)
        return cfg
    
    def validate_prefill_worker(prefill_worker: Service):
        assert prefill_worker.extraPodSpec is not None
        assert prefill_worker.extraPodSpec.mainContainer is not None

        args = break_arguments(prefill_worker.extraPodSpec.mainContainer.args)
        assert "--disaggregation-mode" not in args
        assert "prefill" not in args
        assert "--disaggregation-strategy" not in args
        assert "decode_first" not in args
        #TODO: assert rest of arg translation

        assert prefill_worker.replicas == 1

    cfg = get_prefill_config(trtllm_config)
    assert cfg.spec.services.get("Planner") is None
    assert get_service_name_from_sub_component_type(cfg.spec.services, "decode") is None
    prefill_worker_name = get_service_name_from_sub_component_type(cfg.spec.services, "prefill")
    assert prefill_worker_name is not None
    prefill_worker = cfg.spec.services.get(prefill_worker_name)
    assert prefill_worker is not None
    validate_prefill_worker(prefill_worker)

    cfg = get_prefill_config(trtllm_config_no_sub_component_type)
    assert cfg.spec.services.get("Planner") is None
    prefill_worker = cfg.spec.services.get("TRTLLMWorker")
    assert prefill_worker is not None
    assert cfg.spec.services.get("TRTLLMPrefillWorker") is None
    assert cfg.spec.services.get("TRTLLMDecodeWorker") is None
    validate_prefill_worker(prefill_worker)

def test_trtllm_config_convert_config_decode(trtllm_config, trtllm_config_no_sub_component_type):
    config_modifier = TrtllmConfigModifier()

    def get_decode_config(config: dict) -> Config: 
        decode_config = config_modifier.convert_config(config, "decode")
        cfg = Config.model_validate(decode_config)
        return cfg
    
    def validate_decode_worker(decode_worker: Service):
        assert decode_worker.extraPodSpec is not None
        assert decode_worker.extraPodSpec.mainContainer is not None

        args = break_arguments(decode_worker.extraPodSpec.mainContainer.args)
        assert "--disaggregation-mode" not in args
        assert "prefill" not in args
        assert "--disaggregation-strategy" not in args
        assert "decode_first" not in args
        #TODO: assert rest of arg translation

        assert decode_worker.replicas == 1

    cfg = get_decode_config(trtllm_config)
    assert cfg.spec.services.get("Planner") is None
    assert get_service_name_from_sub_component_type(cfg.spec.services, "prefill") is None
    decode_worker_name = get_service_name_from_sub_component_type(cfg.spec.services, "decode")
    assert decode_worker_name is not None
    decode_worker = cfg.spec.services.get(decode_worker_name)
    assert decode_worker is not None
    validate_decode_worker(decode_worker)

    cfg = get_decode_config(trtllm_config_no_sub_component_type)
    assert cfg.spec.services.get("Planner") is None
    decode_worker = cfg.spec.services.get("TRTLLMWorker")
    assert decode_worker is not None
    assert cfg.spec.services.get("TRTLLMPrefillWorker") is None
    assert cfg.spec.services.get("TRTLLMDecodeWorker") is None
    validate_decode_worker(decode_worker)

def test_vllm_config_set_config_tp_size(vllm_config, vllm_config_no_sub_component_type):
    config_modifier = VllmV1ConfigModifier()

    def validate_worker(worker: Service, tp_size: int):
        assert worker.resources is not None
        assert worker.resources.requests is not None
        assert worker.resources.requests["gpu"] == str(tp_size)
        assert worker.resources.limits is not None
        assert worker.resources.limits["gpu"] == str(tp_size)

        assert worker.extraPodSpec is not None
        assert worker.extraPodSpec.mainContainer is not None

        args = break_arguments(worker.extraPodSpec.mainContainer.args)
        assert "--tensor-parallel-size" in args
        assert args[args.index("--tensor-parallel-size") + 1] == str(tp_size)

    prefill_config = config_modifier.convert_config(vllm_config, "prefill")
    prefill_config = config_modifier.set_config_tp_size(prefill_config, 2, "prefill")
    prefill_cfg = Config.model_validate(prefill_config)

    prefill_worker_name = get_service_name_from_sub_component_type(prefill_cfg.spec.services, "prefill")
    assert prefill_worker_name is not None
    prefill_worker = prefill_cfg.spec.services.get(prefill_worker_name)
    assert prefill_worker is not None
    validate_worker(prefill_worker, 2)

    decode_config = config_modifier.convert_config(vllm_config, "decode")
    decode_config = config_modifier.set_config_tp_size(decode_config, 8, "decode")
    decode_cfg = Config.model_validate(decode_config)

    decode_worker_name = get_service_name_from_sub_component_type(decode_cfg.spec.services, "decode")
    assert decode_worker_name is not None
    decode_worker = decode_cfg.spec.services.get(decode_worker_name)
    assert decode_worker is not None
    validate_worker(decode_worker, 8)

    prefill_config = config_modifier.convert_config(vllm_config_no_sub_component_type, "prefill")
    prefill_config = config_modifier.set_config_tp_size(prefill_config, 4, "prefill")
    prefill_cfg = Config.model_validate(prefill_config)
    prefill_worker = prefill_cfg.spec.services.get("VllmDecodeWorker")
    assert prefill_worker is not None
    validate_worker(prefill_worker, 4)

    decode_config = config_modifier.convert_config(vllm_config_no_sub_component_type, "decode")
    decode_config = config_modifier.set_config_tp_size(decode_config, 16, "decode")
    decode_cfg = Config.model_validate(decode_config)
    decode_worker = decode_cfg.spec.services.get("VllmDecodeWorker")
    assert decode_worker is not None
    validate_worker(decode_worker, 16)

def test_sglang_config_set_config_tp_size(sglang_config, sglang_config_no_sub_component_type):
    config_modifier = SGLangConfigModifier()

    def validate_worker(worker: Service, tp_size: int):
        assert worker.resources is not None
        assert worker.resources.requests is not None
        assert worker.resources.requests["gpu"] == str(tp_size)
        assert worker.resources.limits is not None
        assert worker.resources.limits["gpu"] == str(tp_size)

        assert worker.extraPodSpec is not None
        assert worker.extraPodSpec.mainContainer is not None

        args = break_arguments(worker.extraPodSpec.mainContainer.args)
        assert "--tp" in args
        assert args[args.index("--tp") + 1] == str(tp_size)

    prefill_config = config_modifier.convert_config(sglang_config, "prefill")
    prefill_config = config_modifier.set_config_tp_size(prefill_config, 2, "prefill")
    prefill_cfg = Config.model_validate(prefill_config)

    prefill_worker_name = get_service_name_from_sub_component_type(prefill_cfg.spec.services, "prefill")
    assert prefill_worker_name is not None
    prefill_worker = prefill_cfg.spec.services.get(prefill_worker_name)
    assert prefill_worker is not None
    validate_worker(prefill_worker, 2)

    decode_config = config_modifier.convert_config(sglang_config, "decode")
    decode_config = config_modifier.set_config_tp_size(decode_config, 8, "decode")
    decode_cfg = Config.model_validate(decode_config)

    decode_worker_name = get_service_name_from_sub_component_type(decode_cfg.spec.services, "decode")
    assert decode_worker_name is not None
    decode_worker = decode_cfg.spec.services.get(decode_worker_name)
    assert decode_worker is not None
    validate_worker(decode_worker, 8)

    prefill_config = config_modifier.convert_config(sglang_config_no_sub_component_type, "prefill")
    prefill_config = config_modifier.set_config_tp_size(prefill_config, 4, "prefill")
    prefill_cfg = Config.model_validate(prefill_config)
    prefill_worker = prefill_cfg.spec.services.get("SGLangDecodeWorker")
    assert prefill_worker is not None
    validate_worker(prefill_worker, 4)

    decode_config = config_modifier.convert_config(sglang_config_no_sub_component_type, "decode")
    decode_config = config_modifier.set_config_tp_size(decode_config, 16, "decode")
    decode_cfg = Config.model_validate(decode_config)
    decode_worker = decode_cfg.spec.services.get("SGLangDecodeWorker")
    assert decode_worker is not None
    validate_worker(decode_worker, 16)

def test_trtllm_config_set_config_tp_size(trtllm_config, trtllm_config_no_sub_component_type):
    config_modifier = TrtllmConfigModifier()

    def validate_worker(worker: Service, tp_size: int):
        assert worker.resources is not None
        assert worker.resources.requests is not None
        assert worker.resources.requests["gpu"] == str(tp_size)
        assert worker.resources.limits is not None
        assert worker.resources.limits["gpu"] == str(tp_size)

        assert worker.extraPodSpec is not None
        assert worker.extraPodSpec.mainContainer is not None

        args = break_arguments(worker.extraPodSpec.mainContainer.args)
        override_dict, args = parse_override_engine_args(args)
        assert override_dict["tensor_parallel_size"] == tp_size

    prefill_config = config_modifier.convert_config(trtllm_config, "prefill")
    prefill_config = config_modifier.set_config_tp_size(prefill_config, 2, "prefill")
    prefill_cfg = Config.model_validate(prefill_config)

    prefill_worker_name = get_service_name_from_sub_component_type(prefill_cfg.spec.services, "prefill")
    assert prefill_worker_name is not None
    prefill_worker = prefill_cfg.spec.services.get(prefill_worker_name)
    assert prefill_worker is not None
    validate_worker(prefill_worker, 2)

    decode_config = config_modifier.convert_config(trtllm_config, "decode")
    decode_config = config_modifier.set_config_tp_size(decode_config, 8, "decode")
    decode_cfg = Config.model_validate(decode_config)

    decode_worker_name = get_service_name_from_sub_component_type(decode_cfg.spec.services, "decode")
    assert decode_worker_name is not None
    decode_worker = decode_cfg.spec.services.get(decode_worker_name)
    assert decode_worker is not None
    validate_worker(decode_worker, 8)

    prefill_config = config_modifier.convert_config(trtllm_config_no_sub_component_type, "prefill")
    prefill_config = config_modifier.set_config_tp_size(prefill_config, 4, "prefill")
    prefill_cfg = Config.model_validate(prefill_config)
    prefill_worker = prefill_cfg.spec.services.get("TRTLLMWorker")
    assert prefill_worker is not None
    validate_worker(prefill_worker, 4)

    decode_config = config_modifier.convert_config(trtllm_config_no_sub_component_type, "decode")
    decode_config = config_modifier.set_config_tp_size(decode_config, 16, "decode")
    decode_cfg = Config.model_validate(decode_config)
    decode_worker = decode_cfg.spec.services.get("TRTLLMWorker")
    assert decode_worker is not None
    validate_worker(decode_worker, 16)