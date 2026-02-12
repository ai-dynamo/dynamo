# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def _try_import_aiconfigurator():
    # Lazy-import aiconfigurator because it's an optional dependency in profile_sla.py
    import aiconfigurator.sdk.backends.factory
    import aiconfigurator.sdk.config
    import aiconfigurator.sdk.inference_session
    import aiconfigurator.sdk.models
    import aiconfigurator.sdk.perf_database

    return aiconfigurator


class AIConfiguratorPerfEstimator:
    """
    This class is used to estimate the performance of a model using aiconfigurator.
    An instance of this class stores information about the model, system, and backend.
    Methods can be called to estimate prefill and/or decode perf for a given ISL, OSL,
    batch_size, and parallelism config.
    """

    def __init__(
        self,
        hf_id: str,  # e.g. "Qwen/Qwen3-32B"
        system: str,  # e.g. "h200_sxm"
        backend: str,  # e.g. "trtllm"
        version: str,  # e.g. "0.20.0"
    ):
        aiconfigurator = _try_import_aiconfigurator()

        logger.info("Loading aiconfigurator database. This might take a few seconds...")
        if not version:
            version = aiconfigurator.sdk.perf_database.get_latest_database_version(
                system,
                backend,
            )
        self.database = aiconfigurator.sdk.perf_database.get_database(
            system=system,
            backend=backend,
            version=version,
        )
        if not self.database:
            raise ValueError(
                f"Database not found for system: {system}, backend: {backend}, version: {version}"
            )
        logger.info("aiconfigurator database loaded.")

        self.backend = aiconfigurator.sdk.backends.factory.get_backend(backend)
        self.hf_id = hf_id

    def _get_model(self, **model_config_kwargs):
        aiconfigurator = _try_import_aiconfigurator()

        # NOTE: MOE models error out unless moe_tp_size and moe_ep_size are provided.
        model_config = aiconfigurator.sdk.config.ModelConfig(**model_config_kwargs)
        model = aiconfigurator.sdk.models.get_model(
            self.hf_id, model_config, self.backend
        )
        return model

    def estimate_perf(
        self,
        isl: int,
        osl: int,
        batch_size: int,
        mode: str = "full",
        **model_config_kwargs,
    ) -> dict[str, Any]:
        """
        Estimate the perf of this model + system + backend + ISL/OSL/model_config
        using aiconfigurator.

        Args:
            isl: Input sequence length
            osl: Output sequence length
            batch_size: Batch size
            mode: Indicates what perf data to estimate.
                "full": Estimate prefill and decode perf.
                "prefill": Only estimate context perf.
                "decode": Only estimate decode perf.
            **model_config_kwargs: aiconfigurator model config kwargs
                                   (such as tp_size, moe_tp_size, etc).

        Returns:
            dict: Perf metrics returned by aiconfigurator
        """
        aiconfigurator = _try_import_aiconfigurator()

        mode_to_aic_mode = {
            "full": "static",
            "prefill": "static_ctx",
            "decode": "static_gen",
        }
        if mode not in mode_to_aic_mode:
            raise ValueError(
                f"Invalid mode: {mode}. Must be one of {list(mode_to_aic_mode.keys())}."
            )

        self.runtime_config = aiconfigurator.sdk.config.RuntimeConfig(
            batch_size=batch_size,
            beam_width=1,
            isl=isl,
            osl=osl,
        )

        model = self._get_model(**model_config_kwargs)
        session = aiconfigurator.sdk.inference_session.InferenceSession(
            model, self.database, self.backend
        )

        summary = session.run_static(
            mode=mode_to_aic_mode[mode], runtime_config=self.runtime_config, stride=32
        )
        summary_df = summary.get_summary_df()

        # Convert pd.Dataframe to dict since there's only one row
        return summary_df.to_dict(orient="records")[0]

    def estimate_prefill_perf(
        self,
        isl: int,
        **model_config_kwargs,
    ) -> dict[str, Any]:
        """
        Estimate the perf of this model + system + backend + etc assuming it is a prefill worker.
        Args:
            isl: Input sequence length
            **model_config_kwargs: aiconfigurator model config kwargs
                                   (such as tp_size, moe_tp_size, etc).

        Returns:
            dict: Perf metrics returned by aiconfigurator
        """
        return self.estimate_perf(
            isl,
            5,  # small osl
            1,  # concurrency = 1
            mode="prefill",
            **model_config_kwargs,
        )

    def get_max_batch_size(
        self,
        isl: int,
        osl: int,
        **model_config_kwargs,
    ) -> int:
        """
        Estimate the largest batch size that would fit on this GPU.
        Args:
            isl: Input sequence length
            osl: Output sequence length
            **model_config_kwargs: aiconfigurator model config kwargs
                                   (such as tp_size, moe_tp_size, etc).

        Returns:
            int: Estimated largest batch size that will fit on the system.
        """
        model = self._get_model(**model_config_kwargs)

        def get_mem_usage(bs: int):
            # TODO: _get_memory_usage might be underestimating because
            # 1. it doesn't account for runtime buffers
            # 2. it calculates num_tokens = isl*bs which ignores osl
            return self.backend._get_memory_usage(
                model, self.database, bs, 1, isl, osl
            )["total"]

        max_memory_gb = self.database.system_spec["gpu"]["mem_capacity"] / (1024**3)

        bs = 1
        if get_mem_usage(bs) > max_memory_gb:
            # Model does not fit on GPU with the given model config.
            return 0

        # Step 1: find upper bound on batch size.
        while get_mem_usage(bs) < max_memory_gb:
            bs *= 2

        # We know that bs // 2 will fit on GPU but bs will not.
        min_bs = bs // 2
        max_bs = bs

        # Step 2: binary search for max batch size that fits on GPU.
        while min_bs < max_bs:
            test_bs = (min_bs + max_bs) // 2
            if get_mem_usage(test_bs) < max_memory_gb:
                # Because of the +1, the new value of min_bs might not fit on the GPU
                # even though test_bs did fit. So at the end when min_bs and max_bs converge,
                # we need to remember to subtract 1 from the result.
                min_bs = test_bs + 1
            else:
                # max_bs is always a value that doesn't fit on the GPU.
                max_bs = test_bs

        return min_bs - 1  # see comment above

    def get_max_kv_tokens(
        self,
        isl: int,
        osl: int,
        **model_config_kwargs,
    ) -> int:
        """
        Estimate the max number of kv cache tokens that will fit on this GPU
        for the given ISL, OSL, and model config.

        Args:
            isl: Input sequence length
            osl: Output sequence length
            **model_config_kwargs: aiconfigurator model config kwargs
                                   (such as tp_size, moe_tp_size, etc).

        Returns:
            int: Estimated number of KV cache tokens that will fit on the system.
        """
        max_concurrency = self.get_max_batch_size(isl, osl, **model_config_kwargs)
        return max_concurrency * (isl + osl)
