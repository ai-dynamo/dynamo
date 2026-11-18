# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load diff: the state that loading a model creates outside its tensors.

``process_weights_after_loading`` leaves more than tensors behind: kernel
objects, scale floats, backend flags, small CPU tables. A read-only GMS engine
maps the writer's already-processed tensors and cannot run that step again.
The writer records what loading changed so that readers can reproduce it.
"""
