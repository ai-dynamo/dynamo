# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed protected-model bootstrap before vLLM argument construction."""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

from dynamo._core import (
    ModelProtectionError,
    is_protected_model,
    prepare_protected_model,
)
from dynamo.common.utils.namespace import get_worker_namespace

logger = logging.getLogger(__name__)

SUPPORTED_VLLM_SERIES = "0.29."
_CONFIG_ENV = "DYN_MODEL_PROTECTION_CONFIG"
_DISABLED_VLLM_PLUGINS = ""
_MULTIMODAL_CACHE_CONNECTOR = "DynamoMultimodalEmbeddingCacheConnector"
_MULTIMODAL_CACHE_MODULE = (
    "dynamo.vllm.multimodal_utils.multimodal_embedding_cache_connector"
)
_engine_core_policy_installed = False
_FORBIDDEN_OPTIONS = {
    "--config",
    "--draft-model",
    "--ec-transfer-config",
    "--enable-lora",
    "--enable-prompt-adapter",
    "--enable-rl",
    "--headless",
    "--hf-config-path",
    "--hf-overrides",
    "--io-processor-plugin",
    "--logits-processors",
    "--model-class-overrides",
    "--model-loader-extra-config",
    "--model-weights",
    "--speculative-config",
    "--speculative-model",
    "--tokenizer",
    "--trust-remote-code",
    "--weight-transfer-config",
    "--worker-extension-cls",
}
_BOOTSTRAP_OPTIONS = {
    "--model",
    "--model-protection-config",
    "--namespace",
    "--served-model-name",
}


@dataclass
class ProtectionBootstrap:
    argv: list[str]
    session: Any | None = None
    model_path: str | None = None
    namespace: str | None = None
    previous_vllm_plugins: str | None = None

    @property
    def protected(self) -> bool:
        return self.session is not None

    def cleanup(self) -> None:
        if self.session is not None:
            session = self.session
            self.session = None
            try:
                session.cleanup()
            finally:
                try:
                    uninstall_protected_engine_core_policy()
                finally:
                    _restore_vllm_plugins(self.previous_vllm_plugins)


def prepare_model_argv(argv: list[str]) -> ProtectionBootstrap:
    """Stage signed public metadata and rewrite one protected local model path."""
    models = _option_values(argv, "--model")
    protected_models = [
        os.path.abspath(model)
        for model in models
        if is_protected_model(os.path.abspath(model))
    ]
    if not protected_models:
        return ProtectionBootstrap(list(argv))
    if len(models) != 1 or len(protected_models) != 1:
        raise ModelProtectionError("MODEL_PROTECTION_CONFIG_INVALID")
    if any(_has_forbidden_option(argv, option) for option in _FORBIDDEN_OPTIONS):
        raise ModelProtectionError("MODEL_PROTECTION_MODE_UNSUPPORTED")
    if os.environ.get("DYN_SNAPSHOT_CONTROL_DIR"):
        raise ModelProtectionError("MODEL_PROTECTION_MODE_UNSUPPORTED")

    # vLLM treats an empty allowlist as "load no plugins". Freeze it before
    # importing vLLM so protected startup cannot execute installed entry points.
    previous_vllm_plugins = os.environ.get("VLLM_PLUGINS")
    os.environ["VLLM_PLUGINS"] = _DISABLED_VLLM_PLUGINS

    try:
        configs = _option_values(argv, "--model-protection-config")
        if len(configs) > 1:
            raise ModelProtectionError("MODEL_PROTECTION_CONFIG_INVALID")
        config_path = configs[0] if configs else os.environ.get(_CONFIG_ENV)

        namespaces = _option_values(argv, "--namespace")
        if len(namespaces) > 1:
            raise ModelProtectionError("MODEL_PROTECTION_CONFIG_INVALID")
        namespace = get_worker_namespace(namespaces[0] if namespaces else None)
        source = protected_models[0]
        session = prepare_protected_model(source, namespace, config_path)
        if session is None:
            raise ModelProtectionError("SECURE_PACKAGE_INVALID")
    except BaseException:
        _restore_vllm_plugins(previous_vllm_plugins)
        raise

    rewritten = _replace_option(argv, "--model", session.model_path)
    if not _has_option(rewritten, "--served-model-name"):
        rewritten.extend(("--served-model-name", models[0]))
    return ProtectionBootstrap(
        rewritten,
        session,
        session.model_path,
        namespace,
        previous_vllm_plugins,
    )


def validate_protected_engine_args(
    config: Any, bootstrap: ProtectionBootstrap, vllm_version: str
) -> None:
    """Validate the normalized vLLM arguments before TPM key release."""
    if not bootstrap.protected:
        return
    if not vllm_version.startswith(SUPPORTED_VLLM_SERIES):
        raise ModelProtectionError("MODEL_PROTECTION_MODE_UNSUPPORTED")
    if config.namespace != bootstrap.namespace or config.model != bootstrap.model_path:
        raise ModelProtectionError("MODEL_PROTECTION_CONFIG_INVALID")
    if any(
        (
            config.headless,
            config.enable_rl,
            config.embedding_worker,
            config.classify_worker,
            config.realtime,
            config.custom_encoder_class is not None,
            config.custom_jinja_template is not None,
        )
    ):
        raise ModelProtectionError("MODEL_PROTECTION_MODE_UNSUPPORTED")

    mode = getattr(config.disaggregation_mode, "value", config.disaggregation_mode)
    if mode not in (None, "agg"):
        raise ModelProtectionError("MODEL_PROTECTION_MODE_UNSUPPORTED")

    engine = config.engine_args
    if any(
        (
            engine.trust_remote_code,
            engine.enable_lora,
            engine.enable_sleep_mode,
            engine.speculative_config is not None,
            engine.model_loader_extra_config not in (None, {}),
            engine.model_weights not in ("", bootstrap.model_path),
            engine.hf_config_path not in (None, "", bootstrap.model_path),
            engine.io_processor_plugin is not None,
            engine.model_class_overrides not in (None, {}),
            engine.hf_overrides not in (None, {}),
            engine.logits_processors is not None,
            engine.weight_transfer_config is not None,
            engine.worker_extension_cls not in (None, ""),
            engine.scheduler_cls is not None,
            engine.kv_transfer_config is not None,
            getattr(engine, "ec_transfer_config", None) is not None,
            engine.nnodes != 1,
        )
    ):
        raise ModelProtectionError("MODEL_PROTECTION_MODE_UNSUPPORTED")
    if (
        engine.tensor_parallel_size != 1
        or engine.pipeline_parallel_size != 1
        or engine.data_parallel_size != 1
    ):
        raise ModelProtectionError("MODEL_PROTECTION_MODE_UNSUPPORTED")
    executor = engine.distributed_executor_backend
    if executor not in (None, "mp", "multiprocessing"):
        raise ModelProtectionError("MODEL_PROTECTION_MODE_UNSUPPORTED")
    worker_cls = engine.worker_cls
    if worker_cls not in (None, "auto"):
        raise ModelProtectionError("MODEL_PROTECTION_MODE_UNSUPPORTED")
    if _value(engine.load_format) not in ("auto", "safetensors"):
        raise ModelProtectionError("MODEL_PROTECTION_MODE_UNSUPPORTED")
    for path in (engine.model, engine.tokenizer, engine.hf_config_path):
        if path not in (None, "", bootstrap.model_path):
            raise ModelProtectionError("MODEL_PROTECTION_CONFIG_INVALID")


def validate_protected_vllm_config(
    vllm_config: Any, bootstrap: ProtectionBootstrap
) -> None:
    """Recheck paths and topology after vLLM normalization."""
    if not bootstrap.protected:
        return
    model = vllm_config.model_config
    for path in (
        model.model,
        model.model_weights,
        model.tokenizer,
    ):
        if path not in (None, "", bootstrap.model_path):
            raise ModelProtectionError("MODEL_PROTECTION_CONFIG_INVALID")
    parallel = vllm_config.parallel_config
    if (
        parallel.tensor_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
    ):
        raise ModelProtectionError("MODEL_PROTECTION_MODE_UNSUPPORTED")
    if _value(vllm_config.load_config.load_format) not in ("auto", "safetensors"):
        raise ModelProtectionError("MODEL_PROTECTION_MODE_UNSUPPORTED")
    if (
        vllm_config.kv_transfer_config is not None
        or vllm_config.scheduler_config.scheduler_cls is not None
    ):
        raise ModelProtectionError("MODEL_PROTECTION_MODE_UNSUPPORTED")
    _validate_multimodal_cache(
        getattr(vllm_config, "ec_transfer_config", None), bootstrap
    )


def install_protected_engine_core_policy() -> None:
    """Apply the persistence policy inside vLLM's spawned EngineCore process."""
    global _engine_core_policy_installed
    from vllm.v1.engine.core import EngineCoreProc

    if _engine_core_policy_installed:
        return
    setattr(
        EngineCoreProc,
        "_dynamo_model_protection_original",
        EngineCoreProc.run_engine_core,
    )
    setattr(
        EngineCoreProc, "run_engine_core", staticmethod(_protected_engine_core_entry)
    )
    _engine_core_policy_installed = True


def uninstall_protected_engine_core_policy() -> None:
    """Restore vLLM's entrypoint after the protected worker has stopped."""
    global _engine_core_policy_installed
    if not _engine_core_policy_installed:
        return
    from vllm.v1.engine.core import EngineCoreProc

    original = getattr(EngineCoreProc, "_dynamo_model_protection_original")
    setattr(EngineCoreProc, "run_engine_core", staticmethod(original))
    delattr(EngineCoreProc, "_dynamo_model_protection_original")
    _engine_core_policy_installed = False


def _protected_engine_core_entry(*args: Any, **kwargs: Any) -> Any:
    from vllm.v1.engine.core import EngineCoreProc

    from dynamo._core import enforce_model_protection_process_policy

    enforce_model_protection_process_policy()
    target = getattr(
        EngineCoreProc,
        "_dynamo_model_protection_original",
        EngineCoreProc.run_engine_core,
    )
    return target(*args, **kwargs)


async def materialize_protected_weights(
    bootstrap: ProtectionBootstrap, shutdown_event: asyncio.Event
) -> None:
    """Materialize off-loop and stop the writer before shutdown cleanup."""
    if bootstrap.session is None:
        return
    cancellation = bootstrap.session.cancellation()
    materialize = asyncio.create_task(asyncio.to_thread(bootstrap.session.materialize))
    shutdown = asyncio.create_task(shutdown_event.wait())
    try:
        done, _ = await asyncio.wait(
            (materialize, shutdown), return_when=asyncio.FIRST_COMPLETED
        )
        if shutdown in done and not materialize.done():
            cancellation.cancel()
        await asyncio.shield(materialize)
    except BaseException:
        cancellation.cancel()
        while not materialize.done():
            try:
                await asyncio.shield(materialize)
            except asyncio.CancelledError:
                cancellation.cancel()
            except BaseException:
                break
        if materialize.done() and not materialize.cancelled():
            try:
                materialize.result()
            except BaseException:
                logger.debug(
                    "protected weight materialization failed while the caller "
                    "was already unwinding",
                    exc_info=True,
                )
        raise
    finally:
        shutdown.cancel()
        await asyncio.gather(shutdown, return_exceptions=True)


def _option_values(argv: list[str], option: str) -> list[str]:
    values = []
    for index, token in enumerate(argv):
        if token == option and index + 1 < len(argv):
            values.append(argv[index + 1])
        elif token.startswith(f"{option}="):
            values.append(token.split("=", 1)[1])
    return values


def _has_option(argv: list[str], option: str) -> bool:
    return any(token == option or token.startswith(f"{option}=") for token in argv)


def _has_forbidden_option(argv: list[str], option: str) -> bool:
    """Treat an argparse abbreviation of a forbidden protected option as forbidden."""
    return any(
        token.startswith("--")
        and (name := token.split("=", 1)[0]) not in _BOOTSTRAP_OPTIONS
        and option.startswith(name)
        for token in argv
    )


def _replace_option(argv: list[str], option: str, value: str) -> list[str]:
    rewritten = list(argv)
    for index, token in enumerate(rewritten):
        if token == option:
            rewritten[index + 1] = value
            return rewritten
        if token.startswith(f"{option}="):
            rewritten[index] = f"{option}={value}"
            return rewritten
    raise ModelProtectionError("MODEL_PROTECTION_CONFIG_INVALID")


def _value(value: Any) -> str:
    return str(getattr(value, "value", value)).lower()


def _restore_vllm_plugins(previous: str | None) -> None:
    if previous is None:
        os.environ.pop("VLLM_PLUGINS", None)
    else:
        os.environ["VLLM_PLUGINS"] = previous


def _validate_multimodal_cache(config: Any, bootstrap: ProtectionBootstrap) -> None:
    if config is None:
        return
    extra = getattr(config, "ec_connector_extra_config", None)
    if (
        getattr(config, "ec_role", None) != "ec_both"
        or getattr(config, "ec_connector", None) != _MULTIMODAL_CACHE_CONNECTOR
        or getattr(config, "ec_connector_module_path", None) != _MULTIMODAL_CACHE_MODULE
        or not str(getattr(config, "engine_id", "")).startswith(
            f"{bootstrap.namespace}."
        )
        or not str(getattr(config, "engine_id", "")).endswith(".backend.0")
        or not isinstance(extra, dict)
        or set(extra)
        != {"multimodal_embedding_cache_capacity_gb", "component", "model_name"}
        or not isinstance(extra["multimodal_embedding_cache_capacity_gb"], (int, float))
        or isinstance(extra["multimodal_embedding_cache_capacity_gb"], bool)
        or extra["multimodal_embedding_cache_capacity_gb"] <= 0
        or not isinstance(extra["component"], str)
        or not isinstance(extra["model_name"], str)
    ):
        raise ModelProtectionError("MODEL_PROTECTION_MODE_UNSUPPORTED")
