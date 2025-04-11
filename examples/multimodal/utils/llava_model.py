# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM

import torch
import torch.nn as nn

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"
    
def build_vision_projector(config):
    modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
    modules.append(nn.GELU())
    modules.append(nn.Linear(config.hidden_size, config.hidden_size))
    return nn.Sequential(*modules)

class LlavaModel:
    def __init__(self, config):
        super(LlavaModel, self).__init__(config)
        self.config = config
        self.mm_projector = build_vision_projector(config)

class LlavaLlamaModel(LlavaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)

class LlavaForCausalLM(LlamaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        token_num, tokem_dim = self.lm_head.out_features, self.lm_head.in_features
        if self.lm_head.weight.shape[0] != token_num:
            self.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, dtype=self.dtype))
            self.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, dtype=self.dtype))

        # Initialize weights and apply final processing
        self.post_init()
    
    def mm_project(self, image_features: torch.Tensor):
        return self.model.mm_projector(image_features)
