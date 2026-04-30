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


def test_deepseek_reap_workers_keep_indexcache_turboquant_hicache_pd_contract():
    for service_name, mode in (("prefill", "prefill"), ("decode", "decode")):
        args = _args_for(service_name)

        assert _arg_value(args, "--disaggregation-mode") == mode
        assert _arg_value(args, "--disaggregation-transfer-backend") == "nixl"
        assert _arg_value(args, "--tp") == "4"
        assert _arg_value(args, "--dp") == "4"
        assert "--enable-dp-attention" in args
        assert "--enable-hierarchical-cache" in args
        assert "--kv-events-config" in args
        assert _arg_value(args, "--nsa-indexer-mode") == "indexcache"
        assert "--enable-turboquant-dense-kv-cache" in args
        assert _arg_value(args, "--turboquant-dense-kv-preset") == "latent_2p5bit_nc"

        joined = " ".join(args).lower()
        assert "hisa" not in joined
        assert "hisparse" not in joined


def test_deepseek_reap_smc_is_decode_only():
    prefill_args = _args_for("prefill")
    decode_args = _args_for("decode")

    assert "--speculative-algorithm" not in prefill_args
    assert _arg_value(decode_args, "--speculative-algorithm") == "SMC"
    assert "--speculative-draft-model-path" in decode_args
