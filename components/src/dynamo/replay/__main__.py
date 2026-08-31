# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def _load_main():
    try:
        from dynamo.replay.main import main
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing == "aisimulate" or missing.startswith("aisimulate."):
            raise SystemExit(
                "Dynamo replay requires the 'aisimulate' package and currently "
                "supports Python 3.10-3.12."
            ) from exc
        raise
    return main


if __name__ == "__main__":
    raise SystemExit(_load_main()())
