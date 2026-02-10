# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This module previously contained manual vision model loading and encoding
# functions (load_vision_model, encode_image_embeddings, SupportedModels, etc.).
# All of that logic has been replaced by SGLang's MMEncoder._encode() which
# handles vision model loading and feature extraction internally in a
# model-agnostic way.
