# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Startup helpers for the Dynamo TensorRT-LLM backend."""

import logging
import os


def configure_gms_openmpi_defaults() -> None:
    """Keep TRT-LLM's local mpi4py pool off UCX unless explicitly configured."""

    defaults = {
        "OMPI_MCA_pml": "ob1",
        "OMPI_MCA_btl": "self,vader,tcp",
        "OMPI_MCA_coll_hcoll_enable": "0",
    }
    applied = []
    for key, value in defaults.items():
        if key not in os.environ:
            os.environ[key] = value
            applied.append(f"{key}={value}")

    if applied:
        logging.info("Configured TRT-LLM GMS OpenMPI defaults: %s", ", ".join(applied))
