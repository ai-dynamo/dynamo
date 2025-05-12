"""
In SGLang, we must select a pair of prefill/decode workers and send them both 
the request. At some point this can probably be handled in Rust land but for now I'm adding
a simplified version here that simply passes the request through to both workers
"""

from components.worker import SGLangWorker
from components.decode_worker import SGLangDecodeWorker
import random
import json
from utils.protocol import DisaggPreprocessedRequest, PreprocessedRequest
import logging
logger = logging.getLogger(__name__)

from dynamo.llm import ModelType, register_llm
from dynamo.sdk import async_on_start, depends, dynamo_context, dynamo_endpoint, service
from utils.sglang import parse_sglang_args

@service(
    dynamo={
        "namespace": "dynamo",
    },
    workers=1,
)
class SGLangDisaggLoadBalancer:
    sglang_worker = depends(SGLangWorker)

    def __init__(self):
        class_name = self.__class__.__name__
        self.engine_args = parse_sglang_args(class_name, "")
        self._cached_prefill_urls = {}

    @async_on_start
    async def async_init(self):
        runtime = dynamo_context["runtime"]

        comp_ns, comp_name = SGLangWorker.dynamo_address() # type: ignore
        self.prefill_client = await runtime.namespace(comp_ns).component(comp_name).endpoint("generate").client()
        self.prefill_bootstrap_client = await runtime.namespace(comp_ns).component(comp_name).endpoint("get_url").client()

        comp_ns, comp_name = SGLangDecodeWorker.dynamo_address() # type: ignore
        self.decode_client = await runtime.namespace(comp_ns).component(comp_name).endpoint("generate").client()

        # register llm for discovery
        logger.info("Registering LLM for discovery via load balancer for disaggregation")
        comp_ns, comp_name = self.__class__.dynamo_address() # type: ignore
        endpoint = runtime.namespace(comp_ns).component(comp_name).endpoint("generate")
        await register_llm(
            ModelType.Backend,
            endpoint,
            self.engine_args.model_path,
            self.engine_args.served_model_name,
        )

        logger.info("SGLang Disaggregation Load Balancer initialized")

    @dynamo_endpoint()
    async def generate(self, request: PreprocessedRequest):
        prefill_id, decode_id = await self._select_random_disagg_endpoint_pair()

        logger.info(f"Selected prefill worker {prefill_id} and decode worker {decode_id}")

        if prefill_id not in self._cached_prefill_urls:
            logger.info(f"Fetching prefill bootstrap info for {prefill_id}")
            async for response in await self.prefill_bootstrap_client.direct(
                {}, prefill_id
            ):
                bootstrap_info = json.loads(response)
                logger.info(f"Caching bootstrap info: {bootstrap_info}")
                self._cached_prefill_urls[prefill_id] = bootstrap_info
            
        bootstrap_info = self._cached_prefill_urls[prefill_id]
        logger.info(f"Using bootstrap info: {bootstrap_info}")

        hostname = bootstrap_info.get("host")
        port = bootstrap_info.get("port")

        disagg_request = DisaggPreprocessedRequest(
            request=request,
            bootstrap_host=hostname,
            bootstrap_port=port,
            bootstrap_room=self._generate_bootstrap_room(),
        )

        prefill_resp = self.prefill_client.direct(
            disagg_request.model_dump_json(),
            prefill_id,
        )

        output_generator = self.decode_client.direct(
            prefill_resp.model_dump_json(),
            decode_id,
        )

        async for result in output_generator:
            yield result
        
        await prefill_resp

    def _select_random_disagg_endpoint_pair(self):
        prefill_ids = self.worker_client.endpoint_ids()
        decode_ids = self.decode_client.endpoint_ids()
        return random.choice(prefill_ids), random.choice(decode_ids)

    def _generate_bootstrap_room(self):
        return random.randint(0, 2**63 - 1)

