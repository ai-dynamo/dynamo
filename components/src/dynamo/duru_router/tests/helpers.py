# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def work_step(*, p=100, d=0, root="root", agent="agent", phase="prefill", **fields):
    return dict(
        accepted=True,
        root=root,
        agent=agent,
        phase=phase,
        p=p,
        d=d,
        k=p + d,
        generated_tokens=1,
        **fields,
    )
