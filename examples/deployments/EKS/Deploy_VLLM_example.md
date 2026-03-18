# Steps to deploy vLLM example

## 1. Deploy Dynamo Graph

```
cd dynamo/examples/backends/vllm/deploy
vim agg_router.yaml    #under metadata add namespace: dynamo-system and change image to your built base image
kubectl apply -f agg_router.yaml
```

Your pods should be running like below

```
ubuntu@ip-192-168-83-157:~/dynamo/examples/backends/vllm/deploy$ kubectl get pods -A
NAMESPACE      NAME                                                              READY   STATUS    RESTARTS   AGE
dynamo-system   dynamo-platform-dynamo-operator-controller-manager-86795c5f4j4k   2/2     Running   0          4h17m
dynamo-system   dynamo-platform-etcd-0                                            1/1     Running   0          4h17m
dynamo-system   dynamo-platform-nats-0                                            2/2     Running   0          4h17m
dynamo-system   dynamo-platform-nats-box-5dbf45c748-bxqj7                         1/1     Running   0          4h17m
dynamo-system   vllm-agg-router-frontend-79d599bb9c-fg97p                         1/1     Running   0          4m9s
dynamo-system   vllm-agg-router-vllmdecodeworker-787d575485-hrcjp                 1/1     Running   0          4m9s
dynamo-system   vllm-agg-router-vllmdecodeworker-787d575485-zkwdd                 1/1     Running   0          4m9s
```

Test the Deployment

```
kubectl port-forward deployment/vllm-agg-router-frontend 8000:8000 -n dynamo-system
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
    {
        "role": "user",
        "content": "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden."
    }
    ],
    "stream": false,
    "max_tokens": 30
  }'
```

You should output something similar to below

```
{"id":"chatcmpl-bbe52b36-90ed-4479-9872-89e1aa412aa7","choices":[{"index":0,"message":{"content":"<think>\nOkay, so the user wants me to develop a character background for an explorer named someone in Eldoria. The character is part of the","refusal":null,"tool_calls":null,"role":"assistant","function_call":null,"audio":null},"finish_reason":"stop","logprobs":null}],"created":1753417848,"model":"Qwen/Qwen3-0.6B","service_tier":null,"system_fingerprint":null,"object":"chat.completion","usage":{"prompt_tokens":196,"completion_tokens":29,"total_tokens":225,"prompt_tokens_details":null,"completion_tokens_details":null}}
```