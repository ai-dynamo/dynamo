# Dynamo Deployment of vLLM Example on AWS ECS
## 1. EC2 Cluster Setup (for vLLM workloads)
1. Go to AWS ECS console, **Clusters** tab and click on **Create cluster** with name `dynamo-GPU`
2. Input the cluster name and choose **AWS EC2 instances** as the infrastructure. This option will create a cluster with EC2 instances to deploy containers.
3. Choose the ECS-optimized GPU AMI `Amazon Linux 2 (GPU)` (Amazon ECS–optimized), which includes NVIDIA drivers and the Docker GPU runtime out of the box.
4. Choose `g6e.2xlarge` as the **EC2 instance type** and add an `SSH Key pair` so you can log in the instance for debugging purpose. To test with disaggregated serving, we need at least 2 GPUs, so you can choose `g6e.12xlarge` with 4 GPUs
5. Set **Root EBS volume size** as `200`
6. For the networking, use the default settings. Make sure the **security group** has
- an inbound rule which allows "All traffic" from this security group.
- an inbound rule for port 22 and 8000, so that you can ssh into the instance for debugging purpose
7. Select `Turn on` for **Auto-assign public IP** option.
8. Click on **Create** and a cluster will be deployed through cloudformation.

## 2. Fargate Cluster Setup (for ETCD/NATS services)
1. Go to AWS ECS console, **Clusters** tab and click on **Create cluster**
2. Input the cluster name as `dynamo-fargate`
3. Choose **AWS Fargate (serverless)** as the infrastructure
4. For networking, use the same VPC and subnets as the EC2 cluster to ensure connectivity between services
5. For the security group, use the same security group as the EC2 cluster. This automatically allows communication between all services.
6. Ensure outbound rules allow all traffic (default setting) so the Fargate tasks can download container images and communicate externally
7. Click on **Create** to deploy the Fargate cluster

## 3. ETCD/NATS Task Definitions Setup
Add a task for ETCD and NATS services to run on Fargate. A sample task definition JSON is attached.

### 3.1 Create the ecsTaskExecutionRole (Required)
Before creating the task definitions, you need to create the `ecsTaskExecutionRole` IAM role. This role allows ECS to pull container images from registries and write logs to CloudWatch on your behalf.

> [!IMPORTANT]
> If you create task definitions through the AWS Console's step-by-step wizard, this role is created automatically. However, when importing task definitions from JSON (as recommended in this guide), you must create this role manually.

Follow the [AWS documentation on creating the task execution IAM role](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task_execution_IAM_role.html#create-task-execution-role) to create a role named `ecsTaskExecutionRole` with the `AmazonECSTaskExecutionRolePolicy` policy attached.

Based on the task definition, you may need to add Amazon CloudWatch permissions and AWS Secrets Manager permissions to the `ecsTaskExecutionRole`. See details in the [Amazon CloudWatch Logs permissions reference](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/permissions-reference-cwl.html) the [AWS Secrets Manager authentication and access control guide](https://docs.aws.amazon.com/secretsmanager/latest/userguide/auth-and-access.html#auth-and-access_secrets)

> [!NOTE]
> The role ARN will be `arn:aws:iam::<your-account-id>:role/ecsTaskExecutionRole`. Make sure to update `<your-account-id>` in any task definition JSON files with your actual AWS account ID.

### 3.2 Task Definition Configuration
1. ETCD container
- Container name use `etcd`
- Image URL is `bitnamilegacy/etcd` and **Yes** for Essential container
- Container port

|Container port|Protocol|Port name| App protocol|
|-|-|-|-|
|2379|TCP|2379|HTTP|
|2380|TCP|2380|HTTP|
- Environment variable key is `ALLOW_NONE_AUTHENTICATION` and value is `YES`
2. NATS container
- Container name use `nats`
- Image URL is `nats` and **Yes** for Essential container
- Container port

|Container port|Protocol|Port name| App protocol|
|-|-|-|-|
|4222|TCP|4222|HTTP|
|6222|TCP|6222|HTTP|
|8222|TCP|8222|HTTP|
- Docker configuration, add `-js, --trace` in **Command**

## 4. vLLM Task Definitions Setup
> [!Note]
> Ensure you have created the `ecsTaskExecutionRole` as described in section 3.1 before creating these task definitions.

1. Dynamo vLLM Frontend and Decoding Worker Task
This task will create vLLM frontend, processors, routers and a decoding worker.
Please follow steps below to create this task
- Set container name as `dynamo-frontend` and use prebuild [Dynamo container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/vllm-runtime).
- Choose `Amazon EC2 instances` as the **Launch type** with **Task size** `2 vCPU` and `40 GB`memory
- Choose `host` as the Network mode.
- Container name use `dynamo-vLLM-frontend`
- Add your Image URL (You can use the prebuild [Dynamo container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/vllm-runtime)) and **Yes** for Essential container. It can be AWS ECR URL or Nvidia NGC URL. If using NGC URL, please also choose **Private registry authentication** and add your Secret Manager ARN or name.
- Container port

|Container port|Protocol|Port name| App protocol|
|-|-|-|-|
|8000|TCP|8000|HTTP|
- Use `1` GPU for **Resource allocation limits**
- Environment variables settings as below. Will override the `IP_ADDRESS` later.

|Key|Value type|Value|
|-|-|-|
|ETCD_ENDPOINTS|Value|http://IP_ADDRESS:2379|
|NATS_SERVER|Value|nats://IP_ADDRESS:4222|
- Docker configuration
Add `sh,-c` in **Entry point** and `cd examples/backends/vllm && python -m dynamo.frontend --router-mode kv & python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B --enforce-eager` in **Command**

2. Dynamo vLLM PrefillWorker Task
Create the PrefillWorker task same as the frontend worker, except for following changes
- Set container name as `dynamo-prefill`
- No container port mapping
- Docker configuration with command `cd examples/backends/vllm && python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B --enforce-eager --is-prefill-worker`

## 5. Task Deployment
You can create a service or directly run the task from the task definition
1. ETCD/NATS Task
- Choose the Fargate cluster (`dynamo-fargate`) for **Existing cluster** created in step 2.
- Select **Launch type** as `FARGATE`
- In the **Networking** section, select the same VPC and subnets used for the EC2 cluster
- For **Security group**, select the same security group used by the EC2 cluster
- Verify that outbound rules allow all traffic for downloading images and external communication
- Wait for this deployment to finish, and get the **Private IP** of this task.
2. Dynamo Frontend and Decoding Worker Task
- Choose the EC2 cluster (`dynamo-GPU`) for **Existing cluster** created in step 1.
- In the **Container Overrides**, use the IP for ETCD/NATS task for the `ETCD_ENDPOINTS` and `NATS_SERVER` values.
- After the deployment, an aggregated serving endpoint is created and you can test it with scripts in step 6.
3. Dynamo PrefillWorker Task
- For disaggregated serving, you can deploy a separate prefill worker on another GPU. Choose the EC2 cluster (`dynamo-GPU`) for **Existing cluster** created in step 1 with at least 2 GPUs ( `g6e.12xlarge` for example)
- In the **Container Overrides**, use the IP for ETCD/NATS task for the `ETCD_ENDPOINTS` and `NATS_SERVER` values.

## 6. Testing
Find the public IP of the dynamo frontend task from the task page. Run following commands to query the endpoint.
```sh
export DYNAMO_IP_ADDRESS=TASK_PUBLIC_IP_ADDRESS
curl http://$DYNAMO_IP_ADDRESS:8000/v1/models
curl http://$DYNAMO_IP_ADDRESS:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
    {
        "role": "user",
        "content": "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden."
    }
    ],
    "stream":false,
    "max_tokens": 30
  }'
```
You should be able to see the responses from the hosted endpoint.