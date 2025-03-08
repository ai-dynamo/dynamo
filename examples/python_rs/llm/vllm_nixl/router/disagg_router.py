from dynamo.llm import DisaggregatedRouter

class PyDisaggregatedRouter():
    def __init__(
        self, 
        runtime,
        served_model_name,
        custom_disagg_router=False,
        max_local_prefill_length=1000,
        max_remote_prefill_cache_hit_ratio=0.5,
    ):
        self.runtime = runtime
        self.served_model_name = served_model_name
        self.max_local_prefill_length = max_local_prefill_length
        self.max_remote_prefill_cache_hit_ratio = max_remote_prefill_cache_hit_ratio
        self.custom_disagg_router = custom_disagg_router

        if not self.custom_disagg_router:
            # TODO: add max_remote_prefill_cache_hit_ratio to rust router
            self.disagg_router = DisaggregatedRouter(
                runtime,
                served_model_name,
                max_local_prefill_length,
            )

    def prefill_remote(self, prompt_length, cache_hit_length=0):
        if self.custom_disagg_router:
            # TODO: add max_remote_prefill_cache_hit_ratio to python router
            return prompt_length > self.max_local_prefill_length
        else:
            return self.disagg_router.prefill_remote(prompt_length, cache_hit_length)