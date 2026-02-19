# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse


class ConfigBase:
    """Base configuration class that allows properties with and without defaults in arbitrary order."""

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        obj = cls.__new__(cls)

        # 1) Set everything provided by argparse
        for k, v in vars(args).items():
            setattr(obj, k, v)

        # 2) Populate annotated defaults from the class (and base classes)
        #    only if not already set by argparse.
        for base in reversed(cls.__mro__):
            anns = getattr(base, "__annotations__", {})
            for name in anns:
                if name.startswith("_"):
                    continue

                # IMPORTANT: only skip if it's already set on the INSTANCE
                if name in obj.__dict__:
                    continue

                # If the class defines a default, materialize it onto the instance
                if name in getattr(base, "__dict__", {}):
                    setattr(obj, name, getattr(base, name))

        return obj

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in sorted(self.__dict__.items()))
        return f"{self.__class__.__name__}({items})"
