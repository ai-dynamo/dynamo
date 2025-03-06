import asyncio
import ctypes
import os
import subprocess
from ctypes import c_char_p, c_int64, c_uint32
from time import sleep

import pytest

from dynemo.llm import KvIndexer
from dynemo.runtime import DistributedRuntime

pytestmark = pytest.mark.pre_merge


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    # Setup code
    nats_server = subprocess.Popen(["nats-server", "-js"])
    etcd = subprocess.Popen(["etcd"])
    print("Setting up resources")

    sleep(5)  # wait for nats-server and etcd to start
    yield

    # Teardown code
    print("Tearing down resources")
    nats_server.terminate()
    nats_server.wait()
    etcd.terminate()
    etcd.wait()


async def test_event_handler():
    loop = asyncio.get_running_loop()
    runtime = DistributedRuntime(loop)

    # publisher
    worker_id = 233
    event_publisher = EventPublisher("dynemo", "vllm", worker_id)

    # indexer
    kv_listener = runtime.namespace("dynemo").component("vllm")
    await kv_listener.create_service()
    indexer = KvIndexer(kv_listener)

    test_token = [3] * 64
    lora_id = 0  # lora_id is not used in the indexer
    scores = await indexer.find_matches_for_request(test_token, lora_id)
    assert not scores.scores

    event_publisher.store_event(test_token, lora_id)
    # wait for the event to be processed
    await asyncio.sleep(1)
    scores = await indexer.find_matches_for_request(test_token, lora_id)
    assert scores.scores
    assert worker_id in scores.scores
    assert scores.scores[worker_id] == 1

    # [TODO] remove event


# KV events
class DynemoResult:
    OK = 0
    ERR = 1


class EventPublisher:
    def __init__(self, namespace: str, component: str, worker_id: int):
        self.event_id_counter = 0

        # load event publisher library
        self.lib = ctypes.CDLL(os.environ["VLLM_KV_CAPI_PATH"])
        self.lib.dynemo_llm_init.argtypes = [c_char_p, c_char_p, c_int64]
        self.lib.dynemo_llm_init.restype = c_uint32
        result = self.lib.dynemo_llm_init(
            namespace.encode(), component.encode(), worker_id
        )
        assert result == DynemoResult.OK
        self.lib.dynemo_kv_event_publish_stored.argtypes = [
            ctypes.c_uint64,  # event_id
            ctypes.POINTER(ctypes.c_uint32),  # token_ids
            ctypes.POINTER(ctypes.c_size_t),  # num_block_tokens
            ctypes.POINTER(ctypes.c_uint64),  # block_ids
            ctypes.c_size_t,  # num_blocks
            ctypes.POINTER(ctypes.c_uint64),  # parent_hash
            ctypes.c_uint64,  # lora_id
        ]
        self.lib.dynemo_kv_event_publish_stored.restype = (
            ctypes.c_uint32
        )  # dynemo_llm_result_t

        self.lib.dynemo_kv_event_publish_removed.argtypes = [
            ctypes.c_uint64,  # event_id
            ctypes.POINTER(ctypes.c_uint64),  # block_ids
            ctypes.c_size_t,  # num_blocks
        ]
        self.lib.dynemo_kv_event_publish_removed.restype = (
            ctypes.c_uint32
        )  # dynemo_llm_result_t

    def store_event(self, tokens, lora_id):
        parent_hash = (
            (ctypes.c_uint64 * 1)(self.event_id_counter)
            if self.event_id_counter > 0
            else None
        )
        result = self.lib.dynemo_kv_event_publish_stored(
            self.event_id_counter,  # uint64_t event_id
            (ctypes.c_uint32 * len(tokens))(*tokens),  # const uint32_t *token_ids
            (ctypes.c_size_t * 1)(len(tokens)),  # const uintptr_t *num_block_tokens
            (ctypes.c_uint64 * 1)(self.event_id_counter),  # const uint64_t *block_ids
            1,  # uintptr_t num_blocks
            parent_hash,  # const uint64_t *parent_hash
            lora_id,  # uint64_t lora_id
        )
        self.event_id_counter += 1

        assert result == DynemoResult.OK
