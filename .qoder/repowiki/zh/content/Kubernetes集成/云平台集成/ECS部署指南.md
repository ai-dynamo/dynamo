# ECS部署指南

<cite>
**本文档引用的文件**
- [README.md](file://examples/deployments/ECS/README.md)
- [task_definition_frontend.json](file://examples/deployments/ECS/task_definition_frontend.json)
- [task_definition_etcd_nats.json](file://examples/deployments/ECS/task_definition_etcd_nats.json)
- [task_definition_prefillworker.json](file://examples/deployments/ECS/task_definition_prefillworker.json)
- [dgd.yaml](file://deploy/discovery/dgd.yaml)
- [tuning.md](file://docs/performance/tuning.md)
- [autoscaling.md](file://docs/kubernetes/autoscaling.md)
- [metrics.rs](file://lib/runtime/src/metrics.rs)
</cite>

## 目录
1. [简介](#简介)
2. [项目结构](#项目结构)
3. [核心组件](#核心组件)
4. [架构概览](#架构概览)
5. [详细组件分析](#详细组件分析)
6. [依赖关系分析](#依赖关系分析)
7. [性能考虑](#性能考虑)
8. [故障排查指南](#故障排查指南)
9. [结论](#结论)
10. [附录](#附录)

## 简介

本指南基于Dynamo项目的ECS部署经验，提供完整的AWS ECS (Elastic Container Service) 部署方案。该指南涵盖了ECS集群创建、任务定义配置、服务发现、负载均衡设置，以及与Fargate和EC2的不同部署模式。

Dynamo项目展示了如何在AWS ECS上部署vLLM推理服务，包括ETCD/NATS基础设施服务和Dynamo vLLM工作负载。该部署方案支持单节点和分布式多节点部署，适用于生产环境的高性能推理需求。

## 项目结构

基于仓库中的ECS部署相关文件，项目采用模块化组织方式：

```mermaid
graph TB
subgraph "ECS部署模块"
A[ECS部署文档] --> B[集群配置]
A --> C[任务定义]
A --> D[网络配置]
B --> B1[EC2集群]
B --> B2[Fargate集群]
C --> C1[前端任务]
C --> C2[预填充任务]
C --> C3[基础设施任务]
D --> D1[安全组]
D --> D2[VPC配置]
D --> D3[负载均衡]
end
subgraph "Dynamo组件"
E[Dynamo前端] --> F[vLLM解码器]
E --> G[路由器]
H[Dynamo预填充] --> F
end
C --> E
C --> H
```

**图表来源**
- [README.md](file://examples/deployments/ECS/README.md#L1-L129)
- [task_definition_frontend.json](file://examples/deployments/ECS/task_definition_frontend.json#L1-L79)

**章节来源**
- [README.md](file://examples/deployments/ECS/README.md#L1-L129)

## 核心组件

### ECS集群类型对比

| 组件 | EC2集群 | Fargate集群 |
|------|---------|-------------|
| **计算模式** | 容器直接运行在EC2实例上 | 无服务器容器编排 |
| **GPU支持** | 原生GPU实例支持 | 通过EC2实例类型支持 |
| **启动时间** | 几分钟到十几分钟 | 秒级启动 |
| **成本模型** | 按实例小时付费 | 按秒付费 |
| **适用场景** | 长期稳定工作负载 | 短时突发工作负载 |

### 任务定义配置

每个任务定义包含以下关键配置：

```mermaid
flowchart TD
A[任务定义] --> B[容器配置]
A --> C[资源限制]
A --> D[网络设置]
A --> E[日志配置]
B --> B1[镜像URL]
B --> B2[入口点]
B --> B3[命令参数]
B --> B4[环境变量]
C --> C1[CPU限制]
C --> C2[内存限制]
C --> C3[GPU分配]
D --> D1[网络模式]
D --> D2[端口映射]
D --> D3[安全组]
E --> E1[AWS CloudWatch]
E --> E2[日志驱动]
E --> E3[日志选项]
```

**图表来源**
- [task_definition_frontend.json](file://examples/deployments/ECS/task_definition_frontend.json#L1-L79)
- [task_definition_etcd_nats.json](file://examples/deployments/ECS/task_definition_etcd_nats.json#L1-L112)

**章节来源**
- [task_definition_frontend.json](file://examples/deployments/ECS/task_definition_frontend.json#L1-L79)
- [task_definition_etcd_nats.json](file://examples/deployments/ECS/task_definition_etcd_nats.json#L1-L112)
- [task_definition_prefillworker.json](file://examples/deployments/ECS/task_definition_prefillworker.json#L1-L71)

## 架构概览

Dynamo在ECS上的整体架构设计体现了微服务分离和高可用性原则：

```mermaid
graph TB
subgraph "客户端层"
Client[客户端应用]
end
subgraph "负载均衡层"
ALB[Application Load Balancer]
NLB[NLB - 用于特定场景]
end
subgraph "ECS集群层"
subgraph "EC2集群 (GPU实例)"
EC2Frontend[dynamo-vLLM-frontend]
EC2Prefill[dynamo-prefill]
end
subgraph "Fargate集群"
FargateEtcd[ETCD服务]
FargateNATS[NATS消息队列]
end
end
subgraph "数据存储层"
ETCD[(ETCD键值存储)]
KVCache[KV缓存]
end
Client --> ALB
ALB --> EC2Frontend
EC2Frontend --> EC2Prefill
EC2Frontend --> FargateEtcd
EC2Frontend --> FargateNATS
EC2Prefill --> FargateEtcd
EC2Prefill --> FargateNATS
FargateEtcd --> ETCD
FargateNATS --> KVCache
```

**图表来源**
- [README.md](file://examples/deployments/ECS/README.md#L1-L129)

### 服务发现机制

Dynamo采用多层服务发现策略：

```mermaid
sequenceDiagram
participant Client as 客户端
participant Frontend as Dynamo前端
participant Prefill as 预填充引擎
participant Etcd as ETCD服务
participant NATS as NATS消息队列
Client->>Frontend : HTTP请求
Frontend->>Etcd : 获取路由配置
Etcd-->>Frontend : 返回服务列表
Frontend->>NATS : 发送处理请求
NATS-->>Prefill : 分发任务
Prefill->>Prefill : 执行推理
Prefill-->>NATS : 返回结果
NATS-->>Frontend : 结果通知
Frontend-->>Client : 返回响应
```

**图表来源**
- [dgd.yaml](file://deploy/discovery/dgd.yaml#L1-L59)

## 详细组件分析

### ETCD任务定义分析

ETCD作为关键的数据存储服务，需要高可用性和持久化能力：

| 配置项 | 值 | 说明 |
|--------|----|------|
| **任务家族** | Dynamo-tasks | 任务分组标识 |
| **容器数量** | 1个 | ETCD主容器 |
| **网络模式** | awsvpc | 使用AWS VPC网络 |
| **CPU分配** | 1024 | 1 vCPU |
| **内存分配** | 3072 | 3GB内存 |
| **启动类型** | FARGATE | 无服务器计算 |
| **端口映射** | 2379, 2380 | ETCD通信端口 |

### NATS任务定义分析

NATS作为消息中间件，支持高并发的消息传递：

| 配置项 | 值 | 说明 |
|--------|----|------|
| **任务家族** | Dynamo-tasks | 任务分组标识 |
| **容器数量** | 1个 | NATS主容器 |
| **网络模式** | awsvpc | 使用AWS VPC网络 |
| **CPU分配** | 1024 | 1 vCPU |
| **内存分配** | 3072 | 3GB内存 |
| **启动类型** | FARGATE | 无服务器计算 |
| **端口映射** | 4222, 6222, 8222 | NATS通信端口 |
| **命令参数** | -js, --trace | 启用JetStream和调试 |

### Dynamo前端任务分析

前端服务负责接收客户端请求和协调后端处理：

| 配置项 | 值 | 说明 |
|--------|----|------|
| **任务家族** | Dynamo-frontend | 任务分组标识 |
| **容器数量** | 1个 | 主要业务容器 |
| **网络模式** | host | 直接使用宿主机网络 |
| **CPU分配** | 2048 | 2 vCPU |
| **内存分配** | 40960 | 40GB内存 |
| **GPU分配** | 1个 | NVIDIA GPU |
| **启动类型** | EC2 | EC2实例计算 |
| **端口映射** | 8000 | HTTP服务端口 |
| **环境变量** | ETCD_ENDPOINTS, NATS_SERVER | 服务发现配置 |

### Dynamo预填充任务分析

预填充任务专门处理上下文编码阶段：

| 配置项 | 值 | 说明 |
|--------|----|------|
| **任务家族** | Dynamo-backend | 任务分组标识 |
| **容器数量** | 1个 | 预填充专用容器 |
| **网络模式** | bridge | 桥接网络模式 |
| **CPU分配** | 2048 | 2 vCPU |
| **内存分配** | 40960 | 40GB内存 |
| **GPU分配** | 1个 | NVIDIA GPU |
| **启动类型** | EC2 | EC2实例计算 |
| **端口映射** | 无 | 仅内部通信 |
| **特殊用途** | 预填充专用 | 优化上下文处理 |

**章节来源**
- [task_definition_etcd_nats.json](file://examples/deployments/ECS/task_definition_etcd_nats.json#L1-L112)
- [task_definition_frontend.json](file://examples/deployments/ECS/task_definition_frontend.json#L1-L79)
- [task_definition_prefillworker.json](file://examples/deployments/ECS/task_definition_prefillworker.json#L1-L71)

## 依赖关系分析

### 任务间依赖关系

```mermaid
graph TD
subgraph "基础设施服务"
Etcd[ETCD服务]
NATS[NATS消息队列]
end
subgraph "业务服务"
Frontend[Dynamo前端]
Prefill[Dynamo预填充]
Decode[Dynamo解码器]
end
subgraph "外部依赖"
Registry[容器镜像仓库]
CloudWatch[CloudWatch日志]
Secrets[Secrets Manager]
end
Etcd --> NATS
NATS --> Frontend
NATS --> Prefill
NATS --> Decode
Frontend --> Prefill
Frontend --> Decode
Frontend --> Registry
Prefill --> Registry
Decode --> Registry
Frontend --> CloudWatch
Prefill --> CloudWatch
Decode --> CloudWatch
Frontend --> Secrets
Prefill --> Secrets
Decode --> Secrets
```

**图表来源**
- [README.md](file://examples/deployments/ECS/README.md#L26-L38)

### 网络依赖分析

ECS部署中的网络依赖关系确保了服务间的可靠通信：

```mermaid
flowchart LR
subgraph "VPC网络"
VPC[VPC虚拟私有云]
Subnet[子网]
SG[安全组]
end
subgraph "网络服务"
ELB[弹性负载均衡]
Route53[Route53 DNS]
InternetGW[互联网网关]
end
subgraph "ECS服务"
EC2Cluster[EC2集群]
FargateCluster[Fargate集群]
end
VPC --> Subnet
Subnet --> SG
SG --> ELB
ELB --> EC2Cluster
ELB --> FargateCluster
Route53 --> ELB
InternetGW --> ELB
```

**图表来源**
- [README.md](file://examples/deployments/ECS/README.md#L8-L21)

**章节来源**
- [README.md](file://examples/deployments/ECS/README.md#L1-L129)

## 性能考虑

### 自动扩缩容策略

基于Dynamo项目的性能调优经验，建议采用混合扩缩容策略：

```mermaid
graph TB
subgraph "扩缩容策略"
A[前端服务 - HPA/CPU]
B[预填充服务 - KEDA/TTFT]
C[解码服务 - KEDA/ITL]
D[混合策略 - Planner]
end
subgraph "监控指标"
E[队列深度]
F[首Token时间]
G[令牌间延迟]
H[并发请求数]
end
A --> E
B --> F
C --> G
D --> H
E --> A
F --> B
G --> C
H --> D
```

**图表来源**
- [autoscaling.md](file://docs/kubernetes/autoscaling.md#L1-L732)

### 性能调优要点

根据Dynamo项目的性能调优指南，关键优化参数包括：

| 参数类别 | 优化目标 | 推荐配置 | 影响范围 |
|----------|----------|----------|----------|
| **并行化映射** | GPU利用率最大化 | TP8或TP8PP2 | 整体吞吐量 |
| **批大小** | 内存使用平衡 | 预填充: 小批<br>解码: 大批 | KV缓存效率 |
| **最大令牌数** | 延迟控制 | 预填充: 大值<br>解码: 中等值 | ITL性能 |
| **块大小** | 缓存命中率 | 128 | 前缀缓存效率 |

### 成本优化策略

```mermaid
flowchart TD
A[成本优化] --> B[实例选择]
A --> C[存储优化]
A --> D[网络优化]
B --> B1[GPU实例类型]
B --> B2[预留实例]
B --> B3[Spot实例]
C --> C1[EBS卷类型]
C --> C2[生命周期策略]
C --> C3[压缩存储]
D --> D1[负载均衡]
D --> D2[数据传输]
D --> D3[CDN加速]
```

**章节来源**
- [tuning.md](file://docs/performance/tuning.md#L1-L149)
- [autoscaling.md](file://docs/kubernetes/autoscaling.md#L1-L732)

## 故障排查指南

### 常见问题诊断

```mermaid
flowchart TD
A[服务不可用] --> B{错误类型}
B --> |启动失败| C[容器日志检查]
B --> |网络问题| D[安全组配置]
B --> |资源不足| E[资源监控]
B --> |配置错误| F[配置验证]
C --> C1[CloudWatch日志]
C1 --> C2[容器状态]
C2 --> C3[启动错误]
D --> D1[入站规则]
D --> D2[出站规则]
D --> D3[跨账户访问]
E --> E1[CPU使用率]
E --> E2[内存使用率]
E --> E3[GPU使用率]
F --> F1[任务定义]
F --> F2[环境变量]
F --> F3[挂载点配置]
```

### 日志收集和分析

Dynamo项目提供了完善的日志收集机制：

| 日志类型 | 收集位置 | 分析工具 | 关键指标 |
|----------|----------|----------|----------|
| **容器日志** | CloudWatch Logs | AWS控制台 | 启动时间、错误码 |
| **应用日志** | 容器标准输出 | Fluentd/Fluent Bit | 请求处理时间 |
| **系统日志** | 实例系统日志 | CloudWatch Logs | 内核错误、OOM事件 |
| **网络日志** | VPC流日志 | Athena查询 | 流量统计、连接异常 |

### 性能监控指标

基于Dynamo项目的监控实现，建议关注以下关键指标：

```mermaid
graph LR
subgraph "前端服务指标"
A[dynamo_frontend_queued_requests]
B[dynamo_frontend_inflight_requests]
C[dynamo_frontend_time_to_first_token_seconds]
D[dynamo_frontend_inter_token_latency_seconds]
end
subgraph "后端服务指标"
E[dynamo_worker_queue_length]
F[dynamo_worker_utilization]
G[dynamo_worker_processing_time]
H[dynamo_worker_error_rate]
end
subgraph "系统指标"
I[node_cpu_utilization]
J[node_memory_utilization]
K[node_disk_io]
L[node_network_throughput]
end
```

**图表来源**
- [metrics.rs](file://lib/runtime/src/metrics.rs#L1378-L1510)

**章节来源**
- [metrics.rs](file://lib/runtime/src/metrics.rs#L1378-L1510)

## 结论

基于Dynamo项目的ECS部署实践，本文提供了完整的AWS ECS部署指南。该方案通过合理的任务分离、网络配置和扩缩容策略，实现了高性能的LLM推理服务部署。

关键成功因素包括：
- **正确的集群选择**：根据工作负载特性选择EC2或Fargate
- **合理的资源配置**：平衡性能和成本
- **完善的监控体系**：实时跟踪服务状态
- **灵活的扩缩容策略**：自动适应流量变化
- **高可用架构设计**：确保服务连续性

通过遵循本指南的最佳实践，可以在AWS ECS上构建稳定、高效、可扩展的LLM推理平台。

## 附录

### 部署步骤摘要

1. **集群准备**
   - 创建EC2 GPU集群用于vLLM工作负载
   - 创建Fargate集群用于ETCD/NATS基础设施

2. **IAM角色配置**
   - 创建ecsTaskExecutionRole
   - 配置CloudWatch和Secrets Manager权限

3. **任务定义部署**
   - 部署ETCD/NATS基础设施任务
   - 部署Dynamo前端和预填充任务

4. **网络配置**
   - 配置VPC和子网
   - 设置安全组规则
   - 配置负载均衡器

5. **监控和告警**
   - 设置CloudWatch告警
   - 配置日志聚合
   - 部署Prometheus监控

### 参考配置文件

- **ETCD任务定义**: [task_definition_etcd_nats.json](file://examples/deployments/ECS/task_definition_etcd_nats.json)
- **前端任务定义**: [task_definition_frontend.json](file://examples/deployments/ECS/task_definition_frontend.json)
- **预填充任务定义**: [task_definition_prefillworker.json](file://examples/deployments/ECS/task_definition_prefillworker.json)
- **Dynamo服务发现**: [dgd.yaml](file://deploy/discovery/dgd.yaml)