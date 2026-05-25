# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import torch


class TinyModel(torch.nn.Module):
    def forward(self, x):
        return x * 2.0 + 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", help="Path to write the AOTInductor .pt2 package")
    args = parser.parse_args()

    model = TinyModel().eval()
    example = (torch.arange(4, dtype=torch.float32).reshape(1, 4),)
    exported = torch.export.export(model, example)
    package = torch._inductor.aoti_compile_and_package(
        exported,
        package_path=args.output,
        inductor_configs={"aot_inductor.package": True},
    )
    print(package)


if __name__ == "__main__":
    main()
