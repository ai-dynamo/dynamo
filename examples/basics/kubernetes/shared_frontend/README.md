# Shared Dynamo Frontend
This folder contains kubernetes manifests to deploy Dynamo frontend component as a standalone DynamoGraphDeploymen (DGD)
and two models.
Frontend is shared across the two models. Frontend is deployed to  dynamo namespace `dynamo`, which is a reserved dynamonvcr.io/nvidia/ai-dynamo/vllm-runtime:0.5.0
namespace name for frontend to observe deployed models across all dynamo namespaces. nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.5.0
A shared PVC is configured to store model checkpoint weights fetched from HF.

1. Install Dynamo k8s platform helm chart
2. Create a K8S secret with your Huggingface token and then render k8s manifests
```sh
export HF_TOKEN=YOUR_HF_TOKEN
kubectl create secret generic hf-token-secret \
    --from-literal=HF_TOKEN=${HF_TOKEN} \
    --namespace ${NAMESPACE}
kubectl apply -f shared_frontend.yaml --namespace ${NAMESPACE}
```
3. Testing the deployment and run benchmarks
After deployment, forward the frontend service to access the API:
```sh
kubectl port-forward svc/frontend-frontend 8000:8000 -n ${NAMESPACE}
```
confirm both deployed models are present in the model listing:
```shnvcr.io/nvidia/ai-dynamo/vllm-runtime:0.5.0
curl localhost:8000/v1/models
```
and use following request to test one of the deployed model
```sh
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
You can also benchmark the performance of the endpoint by [GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html)
