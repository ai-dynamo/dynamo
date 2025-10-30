# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KVBM connector for SGLang integration."""

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
from sglang.srt.configs.model_config import ModelConfig

from dynamo.llm import KvbmLeader
from dynamo.llm.sglang_integration.rust import (
    KvConnectorLeader as RustKvConnectorLeader,
)
from dynamo.runtime import DistributedRuntime

logger = logging.getLogger(__name__)


@dataclass
class StoreMetadata:
    last_node: Any
    token_ids: List[int]
    kv_indices: torch.Tensor
    offset: int


@dataclass
class LoadMetadata:
    token_ids: List[int]
    slot_mapping: torch.Tensor
    offset: int


class KVBMLayerwiseConnector:
    def __init__(
        self,
        sgl_config: ModelConfig,
        page_size: int,
        tp_size: int,
        rank: int,
        k_pool: List[torch.Tensor],
        v_pool: List[torch.Tensor],
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        **kwargs,
    ):
        drt = kwargs.get("drt", None)
        if drt is None:
            self.drt = DistributedRuntime.detached()
        else:
            self.drt = drt

        self.sgl_config = sgl_config

        # TODO(ziqif): change world_size to real world size with pp size
        world_size = tp_size

        leader = KvbmLeader(world_size, drt=self.drt)

        print(f"KvConnectorLeader initialized with rank: {rank}")
        self._leader = RustKvConnectorLeader(rank, self.drt, page_size, leader)

        self.sgl_config = sgl_config
        self.tp_size = tp_size
        self.rank = rank
        self.num_layer = sgl_config.num_hidden_layers

        self.layerwise_retrievers: List[Any] = []
        self.layer_load_layer: List[int] = []
        self.kvcaches = [k_pool, v_pool]
        self.tp_group = tp_group
        self.lookup_id_list: List[str] = []

    def load_kv_layerwise(self, layer_id: int) -> None:
        if len(self.layerwise_retrievers) == 0:
            return

        indices_to_remove = []
        for i in range(len(self.layerwise_retrievers)):
            if self.layer_load_layer[i] == layer_id + 1:
                next(self.layerwise_retrievers[i])
                self.layer_load_layer[i] += 1
                if self.layer_load_layer[i] == self.sgl_config.num_hidden_layers:
                    indices_to_remove.append(i)

        for i in sorted(indices_to_remove, reverse=True):
            del self.layerwise_retrievers[i]
            del self.layer_load_layer[i]
            self.lmcache_engine.lookup_unpin(self.lookup_id_list[i])
            del self.lookup_id_list[i]

        return

    def start_load_kv(self, load_metadata: LoadMetadata) -> int:
        token_ids = torch.tensor(load_metadata.token_ids, dtype=torch.int64).cuda()
        # slot_mapping = load_metadata.slot_mapping.cuda()
        offset = load_metadata.offset

        load_mask = torch.ones_like(token_ids, dtype=torch.bool)
        load_mask[:offset] = False

        retrieve_token_num = self._leader.get_num_new_matched_tokens(token_ids, offset)

        # layerwise_retriever = self.lmcache_engine.retrieve_layer(
        #     token_ids[:retrieve_token_num],
        #     mask=load_mask[:retrieve_token_num],
        #     kvcaches=self.kvcaches,
        #     slot_mapping=slot_mapping[:retrieve_token_num],
        #     sync=False,
        # )

        # next(layerwise_retriever)
        # # Load First Layer
        # next(layerwise_retriever)

        # if retrieve_token_num is None:
        #     return 0

        # self.layerwise_retrievers.append(layerwise_retriever)
        # self.layer_load_layer.append(1)

        # self.lookup_id_list.append(lookup_id)

        return retrieve_token_num - offset

    def store_kv(self, store_metadata: StoreMetadata) -> None:
        # TODO(ziqif): implement store_kv
        pass
        # slot_mapping = store_metadata.kv_indices.to(torch.int64).cuda()
        # token_ids = torch.tensor(store_metadata.token_ids, dtype=torch.int64).cuda()
        # store_mask = torch.ones_like(token_ids, dtype=torch.bool)

        # lookup_id = str(uuid.uuid4())
        # self.lmcache_engine.lookup(token_ids, lookup_id=lookup_id, pin=True)

        # layerwise_storer = self.lmcache_engine.store_layer(
        #     token_ids,
        #     mask=store_mask,
        #     kvcaches=self.kvcaches,
        #     slot_mapping=slot_mapping,
        #     offset=store_metadata.offset,
        #     sync=False,
        # )
        # next(layerwise_storer)
        # for _ in range(self.sgl_config.num_hidden_layers):
        #     next(layerwise_storer)

        # self.lmcache_engine.lookup_unpin(lookup_id)
