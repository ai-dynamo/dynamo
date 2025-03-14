from disaggregated.kv_router import Router
from disaggregated.processor import Processor
from disaggregated.worker import VllmWorker
from disaggregated.prefill_worker import PrefillWorker
from disaggregated.frontend import Frontend

"""
Monolith
Frontend.link(DecodeWorker).build()
Kv aware monolith
Frontend.link(Proc).link(DecodeWorker).build()
Kv off + Disag on
Frontend.link(DecodeWorker)
(this can also explicityl be Frontend.link(DecodeWorker).link(PrefillWorker).build()
Kv on + Disag On
Frontend.link(Processor) (
"""

# single worker 
# print(Frontend.dependencies)
# print(VllmWorker.dependencies)
# Frontend.link(VllmWorker).build()
# print(Frontend.dependencies)
# print(VllmWorker.dependencies)
# flags -> model 
# infer endpoint 


# For router llmctl http add chat-models deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B dynamo-init.process.chat/completions
# For non-router llmctl http add chat-models deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B dynamo-init.vllm.generate

# remove models with llmctl http remove deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 

# llmctl add chat-models dynamo-init.process.chat/completions

# example 1 - ex1:Frontend --config basic.yaml
# dynamo serve ex1:Frontend --config basic.yaml OR <cli-flags>
# Frontend.link(VllmWorker) # this should unlink everything after the VllmWorker and unlink the prefill_worker
# VllmWorker.unlink("prefill_worker")

# example 2 - kv cache aware router + worker
Frontend.link(Processor).link(Router).link(VllmWorker)
# print(Frontend.dependencies)

# # example 2 - ex1:Processor
# dynamo serve ex1:Frontend --config kv_worker.yaml
# Frontend.link(Processor).link(VllmWorker) # this should 
# VllmWorker.unlink("prefill_worker")

# example 3 - WORKERS=8 ex1:Processor
# VllmWorker.unlink("prefill_worker")

# example 4 - ex1:VllmWorker

# example 5 - ex1:VllmWorker (need flags for tp and env for WORKERS) 

# example 6 - ex1:Processor (need many flags)

# dynamo serve disaggregated.processor:Processor  \
#    --Processor.model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
#    --Processor.tokenizer=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
#    --Processor.block-size=64 \
#    --Processor.max-model-len=16384 \
#    --Processor.router=kv \
#    --Router.min-workers=1 \
#    --Router.block-size=64 \
#    --Router.model-name=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
#    --VllmWorker.remote-prefill=true \
#    --VllmWorker.model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
#    --VllmWorker.enforce-eager=true \
#    --VllmWorker.tensor-parallel-size=1 \
#    --VllmWorker.kv-transfer-config='{"kv_connector": "DynamoNixlConnector"}' \
#    --VllmWorker.block-size=64  \
#    --VllmWorker.max-num-batched-tokens=16384 \
#    --VllmWorker.max-model-len=16384 \
#    --VllmWorker.router=kv \
#    --VllmWorker.enable-prefix-caching=true \
#    --PrefillWorker.model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
#    --PrefillWorker.enforce-eager=true \
#    --PrefillWorker.block-size=64 \
#    --PrefillWorker.max-model-len=16384 \
#    --PrefillWorker.max-num-batched-tokens=16384 \
#    --PrefillWorker.kv-transfer-config='{"kv_connector": "DynamoNixlConnector"}' \
#    --PrefillWorker.cuda-visible-device-offset=1


# dynamo serve disaggregated.processor:Processor  \
#    --Processor.model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
#    --Processor.router=kv \
#    --Router.min-workers=1 \
#    --Router.model-name=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
#    --VllmWorker.remote-prefill=true \
#    --VllmWorker.tensor-parallel-size=1 \
#    --VllmWorker.router=kv \
#    --VllmWorker.enable-prefix-caching=true \
#    --PrefillWorker.cuda-visible-device-offset=1