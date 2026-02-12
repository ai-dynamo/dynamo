# AKS部署指南

<cite>
**本文档引用的文件**
- [AKS部署文档](file://examples/deployments/AKS/AKS-deployment.md)
- [平台Helm值配置](file://deploy/helm/charts/platform/values.yaml)
- [平台Helm图表定义](file://deploy/helm/charts/platform/Chart.yaml)
- [CRD Helm图表定义](file://deploy/helm/charts/crds/Chart.yaml)
- [安装指南](file://docs/kubernetes/installation_guide.md)
- [Kubernetes部署总览](file://docs/kubernetes/README.md)
- [预部署检查脚本](file://deploy/pre-deployment/pre-deployment-check.sh)
- [GPU库存工具](file://deploy/utils/gpu_inventory.py)
- [基准测试资源设置脚本](file://deploy/utils/setup_benchmarking_resources.sh)
- [服务发现配置](file://deploy/discovery/dgd.yaml)
- [Prometheus监控配置](file://deploy/observability/prometheus.yml)
- [Grafana数据源配置](file://deploy/observability/grafana-datasources.yml)
</cite>

## 目录
1. [简介](#简介)
2. [项目结构](#项目结构)
3. [核心组件](#核心组件)
4. [架构概览](#架构概览)
5. [详细组件分析](#详细组件分析)
6. [依赖关系分析](#依赖关系分析)
7. [性能考虑](#性能考虑)
8. [故障排除指南](#故障排除指南)
9. [结论](#结论)
10. [附录](#附录)

## 简介

本指南为在Azure Kubernetes Service (AKS)上部署Dynamo平台提供完整的技术文档。Dynamo是一个用于大规模语言模型推理的分布式平台，支持多种后端框架（vLLM、SGLang、TensorRT-LLM）和多种部署模式。

本指南涵盖从AKS集群创建到生产环境部署的完整流程，包括Azure特定的网络配置、存储解决方案、安全策略、RBAC权限设置、网络安全组配置以及负载均衡器使用。同时提供Azure Monitor集成、日志聚合和告警配置，以及多可用区部署、自动扩缩容配置和成本优化策略。

## 项目结构

基于仓库中的AKS部署相关文件，主要涉及以下关键目录和文件：

```mermaid
graph TB
subgraph "AKS部署相关文件"
AKS[AKS部署文档<br/>examples/deployments/AKS/AKS-deployment.md]
Values[平台Helm值配置<br/>deploy/helm/charts/platform/values.yaml]
Chart[平台Helm图表<br/>deploy/helm/charts/platform/Chart.yaml]
CRD[CRD Helm图表<br/>deploy/helm/charts/crds/Chart.yaml]
Install[安装指南<br/>docs/kubernetes/installation_guide.md]
Check[预部署检查<br/>deploy/pre-deployment/pre-deployment-check.sh]
end
subgraph "观测性配置"
Prom[Prometheus配置<br/>deploy/observability/prometheus.yml]
Grafana[Grafana数据源<br/>deploy/observability/grafana-datasources.yml]
Utils[工具脚本<br/>deploy/utils/]
end
AKS --> Values
Values --> Chart
Chart --> CRD
Install --> AKS
Check --> AKS
Utils --> Prom
Utils --> Grafana
```

**图表来源**
- [AKS部署文档](file://examples/deployments/AKS/AKS-deployment.md#L1-L79)
- [平台Helm值配置](file://deploy/helm/charts/platform/values.yaml#L1-L732)
- [平台Helm图表定义](file://deploy/helm/charts/platform/Chart.yaml#L1-L46)

**章节来源**
- [AKS部署文档](file://examples/deployments/AKS/AKS-deployment.md#L1-L79)
- [平台Helm值配置](file://deploy/helm/charts/platform/values.yaml#L1-L732)
- [平台Helm图表定义](file://deploy/helm/charts/platform/Chart.yaml#L1-L46)

## 核心组件

### NVIDIA GPU Operator

Dynamo在AKS环境中通过NVIDIA GPU Operator实现GPU资源管理，该组件负责：

- 自动部署和生命周期管理所有NVIDIA软件组件
- GPU驱动程序管理
- 容器工具包自动化
- 设备插件管理
- 监控工具集成

### Dynamo Kubernetes Operator

平台的核心控制器，负责：

- 自定义资源管理（DynamoGraphDeployment等）
- 工作负载编排和调度
- 状态管理和故障恢复
- 与etcd和NATS的协调

### etcd分布式存储

提供高可用的状态存储：

- 分布式键值存储
- 集群状态持久化
- 一致性保证
- 支持单节点和HA模式

### NATS消息系统

用于组件间通信：

- 消息传递和事件分发
- JetStream持久化消息
- 负载均衡和故障转移
- 监控和调试支持

**章节来源**
- [AKS部署文档](file://examples/deployments/AKS/AKS-deployment.md#L22-L55)
- [平台Helm值配置](file://deploy/helm/charts/platform/values.yaml#L234-L287)
- [平台Helm值配置](file://deploy/helm/charts/platform/values.yaml#L289-L490)

## 架构概览

Dynamo在AKS上的整体架构如下：

```mermaid
graph TB
subgraph "Azure AKS集群"
subgraph "控制面"
API[API服务器]
Scheduler[调度器]
Controller[控制器管理器]
end
subgraph "工作节点"
subgraph "GPU节点池"
GPU1[NVIDIA GPU节点1]
GPU2[NVIDIA GPU节点2]
GPU3[NVIDIA GPU节点3]
end
subgraph "CPU节点池"
CPU1[管理节点]
CPU2[辅助节点]
end
end
end
subgraph "Dynamo平台"
subgraph "核心组件"
Operator[Dynamo Kubernetes Operator]
Etcd[etcd集群]
NATS[NATS消息系统]
end
subgraph "推理组件"
Frontend[前端服务]
Prefill[预填充工作器]
Decode[解码工作器]
Router[路由器]
end
end
subgraph "Azure服务"
LB[Azure Load Balancer]
VNet[Azure虚拟网络]
NSG[网络安全组]
AKV[Azure Key Vault]
end
GPU1 --> Operator
GPU2 --> Operator
GPU3 --> Operator
CPU1 --> Operator
CPU2 --> Operator
Operator --> Etcd
Operator --> NATS
Operator --> Frontend
Frontend --> Prefill
Frontend --> Decode
Frontend --> Router
LB --> Frontend
VNet --> GPU1
VNet --> GPU2
VNet --> GPU3
NSG --> LB
AKV --> Operator
```

**图表来源**
- [平台Helm图表定义](file://deploy/helm/charts/platform/Chart.yaml#L24-L46)
- [平台Helm值配置](file://deploy/helm/charts/platform/values.yaml#L213-L287)

## 详细组件分析

### AKS集群创建和配置

#### GPU节点池配置

在AKS中创建支持GPU的节点池需要特别注意以下配置：

```mermaid
flowchart TD
Start([开始创建AKS集群]) --> CheckQuota[检查Azure配额]
CheckQuota --> CreateCluster[创建AKS集群]
CreateCluster --> AddGPU[添加GPU节点池]
AddGPU --> SkipDriver[跳过GPU驱动安装]
SkipDriver --> InstallGPUOp[安装NVIDIA GPU Operator]
InstallGPUOp --> ValidateGPU[验证GPU功能]
ValidateGPU --> End([集群就绪])
AddGPU --> GPUConfig{
GPU节点池配置
}
GPUConfig --> DriverSkip[跳过驱动安装]
GPUConfig --> DriverInstall[手动安装驱动]
DriverSkip --> InstallGPUOp
DriverInstall --> Error[配置错误]
```

**图表来源**
- [AKS部署文档](file://examples/deployments/AKS/AKS-deployment.md#L14-L21)

#### Azure特定网络配置

```mermaid
graph LR
subgraph "Azure网络层"
VNet[Azure虚拟网络]
Subnet[子网配置]
NSG[网络安全组]
ALB[Azure Load Balancer]
end
subgraph "Kubernetes网络"
Service[Service资源]
Ingress[Ingress控制器]
PodNetwork[Pod网络]
end
subgraph "安全策略"
RBAC[RBAC权限]
PSP[PSP策略]
NetPol[网络策略]
end
VNet --> Subnet
Subnet --> NSG
NSG --> ALB
ALB --> Service
Service --> Ingress
PodNetwork --> Service
RBAC --> Service
PSP --> PodNetwork
NetPol --> PodNetwork
```

**图表来源**
- [平台Helm值配置](file://deploy/helm/charts/platform/values.yaml#L115-L136)

### 存储解决方案

#### Azure Files和Azure Disks集成

Dynamo支持多种存储方案：

```mermaid
erDiagram
STORAGE {
string name PK
string type
string size
string access_mode
string mount_options
}
AZURE_FILES {
string file_share_name
string storage_account
string secret_name
string server_ip
}
AZURE_DISKS {
string disk_name
string disk_size_gb
string caching_type
string managed_identity
}
PERSISTENT_VOLUME {
string pv_name PK
string capacity
string reclaim_policy
string volume_mode
}
PERSISTENT_VOLUME_CLAIM {
string pvc_name PK
string access_modes
string volume_name
string storage_class
}
STORAGE ||--|| PERSISTENT_VOLUME : "绑定"
PERSISTENT_VOLUME ||--o{ PERSISTENT_VOLUME_CLAIM : "被请求"
STORAGE ||--o{ AZURE_FILES : "包含"
STORAGE ||--o{ AZURE_DISKS : "包含"
```

**图表来源**
- [预部署检查脚本](file://deploy/pre-deployment/pre-deployment-check.sh#L61-L120)

#### 存储类配置

平台支持多种存储类配置，包括Azure特定的存储类：

- **默认存储类**: 用于动态卷供应
- **高性能存储类**: 使用Premium SSD
- **成本优化存储类**: 使用Standard SSD或HDD

### 安全策略

#### Azure RBAC权限设置

```mermaid
sequenceDiagram
participant User as 用户
participant AKS as AKS集群
participant RBAC as RBAC控制器
participant API as API服务器
User->>AKS : 请求访问集群
AKS->>RBAC : 验证用户身份
RBAC->>RBAC : 检查用户角色绑定
RBAC->>API : 授权访问请求
API->>AKS : 执行操作
AKS->>User : 返回结果
Note over User,RBAC : 基于Azure AD的认证和授权
```

**图表来源**
- [平台Helm值配置](file://deploy/helm/charts/platform/values.yaml#L151-L212)

#### 网络安全组配置

Azure网络安全组用于控制入站和出站流量：

- **前端服务**: 允许HTTP/HTTPS流量
- **管理端口**: 限制到特定IP范围
- **etcd通信**: 仅允许集群内部通信
- **NATS通信**: 控制消息传递流量

### Helm图表配置

#### 平台部署配置

平台Helm图表提供了丰富的配置选项：

```mermaid
classDiagram
class DynamoPlatform {
+bool enabled
+string natsAddr
+string etcdAddr
+DynamoConfig dynamo
+OperatorConfig dynamo-operator
+EtcdConfig etcd
+NatsConfig nats
}
class DynamoConfig {
+string ingressHostSuffix
+bool virtualServiceSupportsHTTPS
+MetricsConfig metrics
+MPIRunConfig mpiRun
}
class OperatorConfig {
+bool enabled
+WebhookConfig webhook
+LeaderElectionConfig leaderElection
+ImagePullSecrets imagePullSecrets
}
class EtcdConfig {
+bool enabled
+string repository
+string tag
+PersistenceConfig persistence
+int replicaCount
}
class NatsConfig {
+bool enabled
+bool tlsCA.enabled
+JetStreamConfig jetstream
+MonitorConfig monitor
}
DynamoPlatform --> DynamoConfig
DynamoPlatform --> OperatorConfig
DynamoPlatform --> EtcdConfig
DynamoPlatform --> NatsConfig
```

**图表来源**
- [平台Helm值配置](file://deploy/helm/charts/platform/values.yaml#L19-L114)
- [平台Helm值配置](file://deploy/helm/charts/platform/values.yaml#L213-L287)
- [平台Helm值配置](file://deploy/helm/charts/platform/values.yaml#L289-L490)

**章节来源**
- [平台Helm值配置](file://deploy/helm/charts/platform/values.yaml#L1-L732)
- [平台Helm图表定义](file://deploy/helm/charts/platform/Chart.yaml#L24-L46)

### 观测性集成

#### Azure Monitor集成

```mermaid
graph TB
subgraph "Dynamo组件"
Operator[Dynamo Operator]
Frontend[前端服务]
Workers[工作器]
end
subgraph "观测性栈"
Prometheus[Prometheus]
Grafana[Grafana]
Tempo[Tempest]
Loki[Loki]
end
subgraph "Azure服务"
AppInsights[Application Insights]
LogAnalytics[Log Analytics]
MetricAlerts[指标告警]
end
Operator --> Prometheus
Frontend --> Prometheus
Workers --> Prometheus
Prometheus --> Grafana
Prometheus --> Tempo
Prometheus --> Loki
Prometheus --> AppInsights
Grafana --> LogAnalytics
Tempo --> MetricAlerts
Loki --> MetricAlerts
```

**图表来源**
- [Prometheus监控配置](file://deploy/observability/prometheus.yml#L20-L57)
- [Grafana数据源配置](file://deploy/observability/grafana-datasources.yml#L18-L24)

#### 日志聚合和告警配置

平台提供完整的日志和监控配置：

- **日志收集**: 使用Loki进行结构化日志收集
- **指标监控**: 使用Prometheus收集系统和应用指标
- **追踪系统**: 使用Tempo进行分布式追踪
- **可视化**: 使用Grafana创建仪表板
- **告警规则**: 基于阈值和业务指标的告警

**章节来源**
- [Prometheus监控配置](file://deploy/observability/prometheus.yml#L1-L63)
- [Grafana数据源配置](file://deploy/observability/grafana-datasources.yml#L1-L24)

## 依赖关系分析

### Helm图表依赖

```mermaid
graph TD
subgraph "Dynamo平台依赖"
Platform[dynamo-platform] --> Operator[dynamo-operator]
Platform --> NATS[nats]
Platform --> Etcd[bitnami/etcd]
Platform --> Kai[kai-scheduler]
Platform --> Grove[grove-charts]
end
subgraph "外部依赖"
Operator --> NGC[NGC镜像仓库]
NATS --> NATSRepo[NATS Helm仓库]
Etcd --> BitnamiRepo[Bitnami Helm仓库]
Kai --> GHCR[GitHub Container Registry]
Grove --> GHCR
end
subgraph "版本兼容性"
Operator --> OpVersion[v0.7.1]
NATS --> NATSVersion[v1.3.2]
Etcd --> EtcdVersion[v12.0.18]
Kai --> KaiVersion[v0.9.4]
Grove --> GroveVersion[v0.1.0-alpha.3]
end
```

**图表来源**
- [平台Helm图表定义](file://deploy/helm/charts/platform/Chart.yaml#L24-L46)

### 组件间通信

```mermaid
sequenceDiagram
participant Client as 客户端
participant Frontend as 前端服务
participant Router as 路由器
participant Prefill as 预填充工作器
participant Decode as 解码工作器
participant Etcd as etcd
participant NATS as NATS
Client->>Frontend : HTTP请求
Frontend->>Router : 路由决策
Router->>Etcd : 查询服务发现
Etcd-->>Router : 返回工作器列表
Router->>NATS : 发送任务消息
NATS->>Prefill : 分发预填充任务
NATS->>Decode : 分发解码任务
Prefill-->>NATS : 任务完成
Decode-->>NATS : 任务完成
NATS->>Router : 任务状态更新
Router->>Frontend : 组合响应
Frontend-->>Client : 返回结果
```

**图表来源**
- [服务发现配置](file://deploy/discovery/dgd.yaml#L10-L11)

**章节来源**
- [平台Helm图表定义](file://deploy/helm/charts/platform/Chart.yaml#L24-L46)
- [服务发现配置](file://deploy/discovery/dgd.yaml#L1-L59)

## 性能考虑

### 多可用区部署

Dynamo支持跨可用区部署以提高可用性和容错能力：

```mermaid
graph LR
subgraph "可用区A"
A1[节点1]
A2[节点2]
A3[节点3]
end
subgraph "可用区B"
B1[节点1]
B2[节点2]
B3[节点3]
end
subgraph "可用区C"
C1[节点1]
C2[节点2]
C3[节点3]
end
subgraph "负载均衡"
LB[Azure Load Balancer]
end
A1 --> LB
A2 --> LB
A3 --> LB
B1 --> LB
B2 --> LB
B3 --> LB
C1 --> LB
C2 --> LB
C3 --> LB
LB --> A1
LB --> B1
LB --> C1
```

### 自动扩缩容配置

平台支持多种扩缩容策略：

- **HPA (Horizontal Pod Autoscaler)**: 基于CPU和内存使用率
- **VPA (Vertical Pod Autoscaler)**: 动态调整资源请求
- **自定义扩缩容**: 基于业务指标和SLA目标

### 成本优化策略

```mermaid
flowchart TD
Start([开始成本优化]) --> Analyze[分析使用模式]
Analyze --> RightSize[正确尺寸规划]
RightSize --> Spot[使用Spot实例]
RightSize --> Reserved[购买预留实例]
Spot --> Optimize[优化资源利用率]
Reserved --> Optimize
Optimize --> Monitor[持续监控]
Monitor --> Analyze
RightSize --> Scheduling[智能调度]
Scheduling --> Affinity[亲和性规则]
Affinity --> Tolerations[容忍度配置]
```

## 故障排除指南

### 常见问题诊断

#### GPU相关问题

```mermaid
flowchart TD
GPUErr[GPU问题] --> CheckDriver[检查驱动安装]
CheckDriver --> CheckOperator[检查GPU Operator]
CheckOperator --> CheckNodes[检查节点状态]
CheckNodes --> VerifyLabels[验证GPU标签]
VerifyLabels --> Reschedule[重新调度Pod]
CheckDriver --> Reinstall[重新安装驱动]
CheckOperator --> Restart[重启Operator]
CheckNodes --> Recreate[重建节点]
```

#### 网络连接问题

```mermaid
flowchart TD
NetErr[网络问题] --> CheckNSG[检查网络安全组]
CheckNSG --> CheckVNet[检查虚拟网络配置]
CheckVNet --> CheckService[检查Service配置]
CheckService --> CheckIngress[检查Ingress控制器]
CheckNSG --> UpdateRules[更新防火墙规则]
CheckVNet --> FixSubnet[修复子网配置]
CheckService --> DebugService[调试Service]
CheckIngress --> DebugIngress[调试Ingress]
```

#### 存储问题

```mermaid
flowchart TD
StorageErr[存储问题] --> CheckSC[检查StorageClass]
CheckSC --> CheckPVC[检查PersistentVolumeClaim]
CheckPVC --> CheckPV[检查PersistentVolume]
CheckPV --> CheckProvisioner[检查存储提供者]
CheckSC --> UpdateSC[更新StorageClass]
CheckPVC --> BoundPV[绑定PV]
CheckPV --> FixProvisioner[修复存储提供者]
CheckProvisioner --> RecreatePVC[重建PVC]
```

### 预部署检查

平台提供了全面的预部署检查脚本，确保集群满足Dynamo部署要求：

**章节来源**
- [预部署检查脚本](file://deploy/pre-deployment/pre-deployment-check.sh#L1-L284)

## 结论

本指南提供了在Azure Kubernetes Service上部署Dynamo平台的完整技术文档。通过合理配置Azure资源、网络和安全策略，结合平台提供的观测性工具，可以构建一个高可用、可扩展且成本优化的推理服务平台。

关键要点包括：
- 利用NVIDIA GPU Operator简化GPU资源管理
- 通过Helm图表实现标准化部署
- 配置Azure特定的网络和存储解决方案
- 实施全面的观测性和告警机制
- 采用多可用区部署提高可用性
- 应用成本优化策略降低运营成本

## 附录

### 快速部署命令

```bash
# 设置环境变量
export NAMESPACE=dynamo-system
export RELEASE_VERSION=0.x.x

# 创建命名空间
kubectl create namespace ${NAMESPACE}

# 安装CRDs
helm install dynamo-crds dynamo-crds-${RELEASE_VERSION}.tgz --namespace default

# 安装平台
helm install dynamo-platform dynamo-platform-${RELEASE_VERSION}.tgz \
  --namespace ${NAMESPACE} \
  --create-namespace
```

### 监控和调试

```bash
# 检查Pod状态
kubectl get pods -n ${NAMESPACE}

# 查看日志
kubectl logs -n ${NAMESPACE} <pod-name>

# 调试命令
kubectl describe pod -n ${NAMESPACE} <pod-name>
```