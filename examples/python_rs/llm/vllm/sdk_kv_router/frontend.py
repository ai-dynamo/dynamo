from compoundai import NOVA_IMAGE, api, depends, service
from sdk_kv_router.processor import Processor


@service(traffic={"timeout": 10000}, image=NOVA_IMAGE)
class Frontend:
    processor = depends(Processor)

    def __init__(self):
        print("frontend init")

    @api
    async def chat_completion(self, msg: str):
        # Call the generate method
        generator = self.processor.generate(
            {
                "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                "messages": [{"role": "user", "content": msg}],
                "stream": True,
                "max_tokens": 10,
            }
        )

        # Now iterate over the async generator
        async for response in generator:
            print("client response_data:", response)
            yield response
