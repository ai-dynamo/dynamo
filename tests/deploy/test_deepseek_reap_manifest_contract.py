# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml


MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "deploy"
    / "production"
    / "examples"
    / "deepseek-v32-reap-sglang.yaml"
)


def _args_for(service_name: str) -> list[str]:
    manifest = yaml.safe_load(MANIFEST.read_text())
    service = manifest["spec"]["services"][service_name]
    return service["extraPodSpec"]["mainContainer"]["args"]


def _arg_value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_deepseek_reap_frontend_uses_event_backed_kv_routing():
    args = _args_for("Frontend")

    assert _arg_value(args, "--router-mode") == "kv"
    assert "--router-kv-events" in args
    assert "--no-kv-events" not in args
    assert "--no-router-kv-events" not in args


def test_deepseek_reap_workers_keep_combined_indexcache_turboquant_hisparse_pd_contract():
    for service_name, mode in (("prefill", "prefill"), ("decode", "decode")):
        args = _args_for(service_name)

        assert _arg_value(args, "--disaggregation-mode") == mode
        assert _arg_value(args, "--disaggregation-transfer-backend") == "nixl"
        assert _arg_value(args, "--tp") == "4"
        assert _arg_value(args, "--dp") == "4"
        assert "--enable-dp-attention" in args
        assert _arg_value(args, "--quantization") == "compressed-tensors"
        assert "--kv-events-config" in args
        assert _arg_value(args, "--kv-cache-dtype") == "bfloat16"
        assert _arg_value(args, "--nsa-prefill-backend") == "flashmla_sparse"
        assert _arg_value(args, "--nsa-decode-backend") == "flashmla_sparse"

        assert "--enable-hierarchical-cache" not in args
        assert _arg_value(args, "--nsa-indexer-mode") == "indexcache"
        assert _arg_value(args, "--nsa-indexcache-pattern") == (
            "FSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSF"
        )
        assert "--enable-turboquant-dense-kv-cache" in args
        assert _arg_value(args, "--turboquant-dense-kv-preset") == "latent_2p5bit_nc"
        assert _arg_value(args, "--turboquant-execution-mode") == "fused_decode"

    prefill_args = _args_for("prefill")
    decode_args = _args_for("decode")

    assert "--enable-hisparse" not in prefill_args
    assert "--disable-radix-cache" not in prefill_args

    assert "--enable-hisparse" in decode_args
    assert "--disable-radix-cache" in decode_args
    assert "--hisparse-config" in decode_args


def test_deepseek_reap_smc_is_decode_only():
    prefill_args = _args_for("prefill")
    decode_args = _args_for("decode")

    assert "--speculative-algorithm" not in prefill_args
    assert _arg_value(decode_args, "--speculative-algorithm") == "SMC"
    assert (
        _arg_value(decode_args, "--speculative-draft-model-path")
        == "/models/smcsd/GLM-4-9B-0414-FP8-DeepSeekV32-OMP"
    )
    assert _arg_value(decode_args, "--speculative-draft-model-quantization") == "fp8"
    assert _arg_value(decode_args, "--speculative-draft-attention-backend") == "triton"
    assert _arg_value(decode_args, "--smc-draft-kv-cache-dtype") == "fp8_e4m3"
    assert _arg_value(decode_args, "--smc-n-particles") == "4"
    assert _arg_value(decode_args, "--smc-gamma") == "6"
