from typing import List
import logging
import json
import os
import asyncio

from tensorrt_llm._torch.speculative.external_api import APIDrafter

from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
# TODO: remove this
logging.getLogger().setLevel(logging.WARNING)

class DynamoAPIDrafter(APIDrafter):
    """
    Custom Dynamo drafter to support internal Dynamo endpoints instead of only HTTP endpoints.
    """
    def __init__(self, spec_config, runtime: DistributedRuntime):
        super().__init__(spec_config)
        self.client = None
        self.max_draft_len = spec_config.max_draft_len
        # TODO: allow custom etcd connection info to be set in the spec_config
        self.connection_info = {}
    
    async def _create_client(self):
        try:
            # parse endpoint
            endpoint_path = self.endpoint.replace("dyn://", "")
            parts = endpoint_path.split(".")
            if len(parts) != 3:
                raise ValueError(f"Invalid Dynamo endpoint format. Received: {self.endpoint}, but expected: dyn://namespace.component.endpoint")
            namespace, component, endpoint = parts

            # create minimal runtime for client access only
            etcd_endpoints = self.connection_info.get("etcd_endpoints", "localhost:2379")
            os.environ.setdefault("ETCD_ENDPOINTS", etcd_endpoints)
            loop = asyncio.get_event_loop()
            self.runtime = DistributedRuntime(loop, False)

            self.client = await self.runtime.namespace(namespace) \
                            .component(component) \
                            .endpoint(endpoint) \
                            .client()
        except Exception as e:
            logging.error(f"Failed to create client for Dynamo endpoint: {self.endpoint} with error: {e}")
            raise e
    
    async def get_draft_tokens(
        self,
        prefix: list[int],
        request_id: int,
        end_id: int,
        max_sequence_length: int,
    ) -> List[int]:
        if self.endpoint.startswith("dyn://"):
            request_data = {
                "token_ids": prefix,
                "sampling_options": {},
                "stop_conditions": {
                    "max_tokens": self.max_draft_len,
                }
            }
            
            if self.client is None:
                await self._create_client()
                pid = os.getpid()
                print(f"[VERIFIER, PID: {pid}]    Created client to endpoint: {self.endpoint}")
            
            draft_tokens = []
            try:
                pid = os.getpid()
                print(f"[VERIFIER, PID: {pid}]    Sending request to drafter: {request_data['token_ids']}")
                print("-"*100)
                response = await self.client.round_robin(request_data)
                
                async for chunk in response:
                    chunk_data = chunk.data()
                    if chunk_data.get("finish_reason"):
                        break
                    draft_tokens.extend(chunk_data.get("token_ids", []))
                    if len(draft_tokens) >= self.max_draft_len:
                        break
                pid = os.getpid()
                print(f"[VERIFIER, PID: {pid}]    Received tokens from drafter: {draft_tokens}")
                print("-"*100)
                print("*"*100)
                print("-"*100)
                return draft_tokens[:self.max_draft_len]
            except Exception as e:
                logging.error(f"Failed to get draft tokens for Dynamo endpoint: {self.endpoint} with error: {e}")
                raise e
        else:
            raise ValueError(f"Invalid Dynamo endpoint format. Received: {self.endpoint}, but expected: dyn://namespace.component.endpoint")
    