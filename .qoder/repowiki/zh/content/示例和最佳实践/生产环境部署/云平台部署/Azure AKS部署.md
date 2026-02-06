# Azure AKS部署

<cite>
**本文档引用的文件**
- [AKS部署指南](file://examples/deployments/AKS/AKS-deployment.md)
- [平台Helm值配置](file://deploy/helm/charts/platform/values.yaml)
- [Dynamo图部署CRD](file://deploy/helm/charts/crds/templates/nvidia.com_dynamographdeployments.yaml)
- [Dynamo图部署类型定义](file://deploy/operator/api/v1alpha1/dynamographdeployment_types.go)
- [GPU库存检查工具](file://deploy/utils/gpu_inventory.py)
- [Kubernetes工具集](file://deploy/utils/kubernetes.py)
- [发现配置示例](file://deploy/discovery/dgd.yaml)
- [PVC声明模板](file://deploy/utils/manifests/pvc.yaml)
- [PVC访问Pod模板](file://deploy/utils/manifests/pvc-access-pod.yaml)
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

## 简介

本指南提供了在Azure Kubernetes Service (AKS)上部署Dynamo的完整解决方案。Dynamo是一个分布式推理平台，支持多种后端框架（如vLLM、TensorRT-LLM、SGLang），通过Kubernetes原生的Operator模式实现自动化的模型服务编排。

本部署方案涵盖了从基础设施准备到生产环境监控的完整流程，包括GPU节点池配置、存储持久化、网络访问控制以及成本优化策略。

## 项目结构

Dynamo在AKS部署相关的代码组织主要分布在以下目录：

```mermaid
graph TB
subgraph "部署相关文件"
A[examples/deployments/AKS/] --> B[AKS部署指南.md]
C[deploy/] --> D[helm/charts/]
C --> E[operator/]
C --> F[observability/]
C --> G[utils/]
C --> H[discovery/]
end
subgraph "Helm图表"
D --> I[platform/]
D --> J[crds/]
I --> K[values.yaml]
I --> L[templates/]
J --> M[CRD模板]
end
subgraph "工具集"
G --> N[gpu_inventory.py]
G --> O[kubernetes.py]
G --> P[manifests/]
P --> Q[pvc.yaml]
P --> R[pvc-access-pod.yaml]
end
```

**图表来源**
- [AKS部署指南](file://examples/deployments/AKS/AKS-deployment.md#L1-L79)
- [平台Helm值配置](file://deploy/helm/charts/platform/values.yaml#L1-L732)

**章节来源**
- [AKS部署指南](file://examples/deployments/AKS/AKS-deployment.md#L1-L79)
- [平台Helm值配置](file://deploy/helm/charts/platform/values.yaml#L1-L732)

## 核心组件

### NVIDIA GPU Operator

Dynamo通过NVIDIA GPU Operator实现GPU资源的自动化管理：

- **GPU驱动程序**: 自动安装和更新NVIDIA设备驱动
- **容器工具包**: 提供容器运行时的GPU支持
- **设备插件**: 暴露GPU资源给Kubernetes调度器
- **监控工具**: 集成DCGM进行GPU性能监控

### Dynamo Kubernetes Operator

Dynamo的核心控制器负责：

- **资源编排**: 自动创建和管理推理服务的Kubernetes资源
- **状态管理**: 监控服务状态并处理故障恢复
- **配置管理**: 处理DynamoGraphDeployment自定义资源
- **服务发现**: 在Kubernetes集群内自动发现推理服务

### 存储系统

支持多种存储后端以满足不同场景需求：

- **Azure Files**: 支持ReadWriteMany访问模式，适合多副本共享
- **Azure Disks**: 单节点读写，高性能本地存储
- **持久卷声明**: 通过动态供应实现弹性存储管理

**章节来源**
- [AKS部署指南](file://examples/deployments/AKS/AKS-deployment.md#L22-L55)
- [平台Helm值配置](file://deploy/helm/charts/platform/values.yaml#L20-L120)

## 架构概览

Dynamo在AKS上的整体架构如下：

```mermaid
graph TB
subgraph "Azure基础设施"
A[AKS集群] --> B[GPU节点池]
A --> C[虚拟网络]
A --> D[负载均衡器]
end
subgraph "Dynamo平台层"
E[Dynamo Operator] --> F[NATS消息系统]
E --> G[etcd状态存储]
E --> H[自定义资源管理]
end
subgraph "推理服务层"
I[前端服务] --> J[vLLM解码器]
I --> K[vLLM预填充]
I --> L[TensorRT-LLM]
I --> M[SGLang]
end
subgraph "存储层"
N[Azure Files] --> O[PersistentVolume]
P[Azure Disks] --> Q[PersistentVolume]
end
subgraph "监控层"
R[Prometheus] --> S[指标收集]
T[Grafana] --> U[可视化面板]
end
B --> I
F --> E
G --> E
H --> E
O --> I
Q --> I
S --> R
U --> T
```

**图表来源**
- [平台Helm值配置](file://deploy/helm/charts/platform/values.yaml#L234-L490)
- [Dynamo图部署CRD](file://deploy/helm/charts/crds/templates/nvidia.com_dynamographdeployments.yaml#L48-L110)

## 详细组件分析

### AKS集群配置

#### GPU节点池设置

推荐使用NC系列或ND系列VM SKU以获得最佳GPU性能：

```mermaid
flowchart TD
A[创建AKS集群] --> B[配置GPU节点池]
B --> C[选择VM SKU]
C --> D[NC64s_v3<br/>2x A100 80GB]
C --> E[ND96amsr_A100_v4<br/>4x A100 80GB]
C --> F[ND40s_v2<br/>2x V100 32GB]
B --> G[设置节点标签]
G --> H[nvidia.com/gpu.type=A100]
G --> I[nvidia.com/gpu.count=2]
B --> J[启用GPU驱动跳过]
J --> K[由GPU Operator管理驱动]
```

**图表来源**
- [AKS部署指南](file://examples/deployments/AKS/AKS-deployment.md#L14-L20)

#### 虚拟网络配置

```mermaid
graph LR
subgraph "Azure虚拟网络"
A[VNet: 10.0.0.0/8] --> B[子网划分]
B --> C[AKS子网: 10.240.0.0/12]
B --> D[应用子网: 10.0.0.0/24]
B --> E[数据库子网: 10.1.0.0/24]
end
subgraph "网络安全组"
F[NSG-AKS] --> G[允许: 443, 80]
F --> H[拒绝: 0.0.0.0/0]
I[NSG-应用] --> J[允许: 8080, 8081]
I --> K[限制: 仅AKS子网]
end
subgraph "防火墙规则"
L[Azure Firewall] --> M[出站: Docker Hub]
L --> N[出站: Azure API]
L --> O[入站: 受限访问]
end
```

### 存储配置

#### Azure Files持久化

```mermaid
sequenceDiagram
participant Client as 客户端
participant AKS as AKS集群
participant AFS as Azure Files
participant PV as PersistentVolume
participant PVC as PersistentVolumeClaim
Client->>AKS : 创建PVC
AKS->>PV : 动态供应
PV->>AFS : 创建文件共享
PV->>PVC : 绑定PVC
AKS->>Client : 返回存储信息
Client->>AKS : 挂载到Pod
AKS->>AFS : 通过CIFS挂载
```

**图表来源**
- [PVC声明模板](file://deploy/utils/manifests/pvc.yaml#L1-L17)
- [PVC访问Pod模板](file://deploy/utils/manifests/pvc-access-pod.yaml#L36-L40)

#### Azure Disks配置

```mermaid
flowchart TD
A[创建Azure Disk] --> B[格式化文件系统]
B --> C[创建PersistentVolume]
C --> D[绑定到PVC]
D --> E[挂载到Pod]
E --> F[单节点读写]
F --> G[高性能I/O]
```

### 网络配置

#### Load Balancer设置

```mermaid
graph TB
subgraph "外部访问"
A[用户请求] --> B[Azure Load Balancer]
B --> C[Ingress Controller]
C --> D[服务端点]
end
subgraph "内部通信"
E[前端服务] --> F[NATS消息]
E --> G[etcd协调]
F --> H[服务发现]
G --> H
end
subgraph "安全策略"
I[网络策略] --> J[命名空间隔离]
I --> K[端口白名单]
I --> L[流量加密]
end
```

#### Application Gateway集成

```mermaid
sequenceDiagram
participant User as 用户浏览器
participant AGW as Azure Application Gateway
participant LB as 负载均衡器
participant SVC as Kubernetes服务
participant POD as 推理Pod
User->>AGW : HTTPS请求
AGW->>LB : 转发到后端池
LB->>SVC : 路由到适当端点
SVC->>POD : 分发到工作Pod
POD-->>User : 响应数据
Note over AGW,SVC : 支持SSL终止和WAF
```

### 监控和日志

#### Prometheus集成

```mermaid
graph LR
subgraph "监控栈"
A[Prometheus] --> B[指标抓取]
B --> C[NATS指标]
B --> D[etcd指标]
B --> E[GPU指标]
B --> F[应用指标]
G[Grafana] --> H[仪表板]
H --> I[实时监控]
J[AlertManager] --> K[告警通知]
end
subgraph "指标类型"
C --> L[连接数]
D --> M[存储使用率]
E --> N[利用率]
F --> O[延迟分布]
end
```

**图表来源**
- [Prometheus监控配置](file://deploy/observability/prometheus.yml#L20-L50)
- [Grafana数据源配置](file://deploy/observability/grafana-datasources.yml#L18-L24)

**章节来源**
- [Prometheus监控配置](file://deploy/observability/prometheus.yml#L1-L63)
- [Grafana数据源配置](file://deploy/observability/grafana-datasources.yml#L1-L24)

## 依赖关系分析

### 组件依赖图

```mermaid
graph TB
subgraph "基础设施依赖"
A[Azure CLI] --> B[AKS集群]
C[az命令] --> B
D[kubectl] --> B
end
subgraph "软件依赖"
E[Helm 3+] --> F[Dynamo Charts]
G[NVIDIA GPU Operator] --> H[驱动管理]
I[Kubernetes API] --> J[CRD支持]
end
subgraph "运行时依赖"
K[Dynamo Operator] --> L[NATS消息]
K --> M[etcd存储]
K --> N[自定义资源]
O[vLLM/TensorRT-LLM] --> P[GPU加速]
Q[SGLang] --> P
end
subgraph "监控依赖"
R[Prometheus] --> S[指标导出器]
T[Grafana] --> U[数据源]
V[DCGM] --> W[硬件监控]
end
A --> E
C --> G
D --> I
G --> K
I --> K
L --> R
M --> R
N --> R
```

**图表来源**
- [平台Helm值配置](file://deploy/helm/charts/platform/values.yaml#L1-L100)
- [Dynamo图部署CRD](file://deploy/helm/charts/crds/templates/nvidia.com_dynamographdeployments.yaml#L1-L50)

### 资源配额和限制

| 组件 | CPU请求 | 内存请求 | GPU限制 | 存储需求 |
|------|---------|----------|---------|----------|
| Dynamo Operator | 200m | 256Mi | 无 | 无 |
| NATS服务器 | 500m | 1Gi | 无 | 10Gi |
| etcd集群 | 1核 | 2Gi | 无 | 1Gi |
| vLLM解码器 | 2核 | 8Gi | 1个A100 | 50Gi |
| vLLM预填充 | 1核 | 4Gi | 1个A100 | 50Gi |
| TensorRT-LLM | 4核 | 16Gi | 1个A100 | 50Gi |
| SGLang引擎 | 2核 | 8Gi | 1个A100 | 50Gi |

**章节来源**
- [平台Helm值配置](file://deploy/helm/charts/platform/values.yaml#L234-L490)
- [Dynamo图部署类型定义](file://deploy/operator/api/v1alpha1/dynamographdeployment_types.go#L47-L69)

## 性能考虑

### GPU VM SKU选择策略

```mermaid
flowchart TD
A[性能需求评估] --> B{模型大小}
B --> |小模型(<7B)| C[NC64s_v3<br/>2x A100 80GB]
B --> |中等模型(7B-65B)| D[ND40s_v2<br/>2x V100 32GB]
B --> |大模型(>65B)| E[ND96amsr_A100_v4<br/>4x A100 80GB]
C --> F[成本优化]
D --> F
E --> F
F --> G{预算约束}
G --> |充足| H[专用实例]
G --> |有限| I[Spot实例]
H --> J[稳定性能]
I --> K[成本降低]
```

### Spot VM使用策略

```mermaid
sequenceDiagram
participant Scheduler as Kubernetes调度器
participant SpotPool as Spot实例池
participant Node as 工作节点
participant Pod as 推理Pod
Scheduler->>SpotPool : 请求Spot实例
SpotPool->>Scheduler : 分配实例
Scheduler->>Node : 创建节点
Node->>Pod : 启动推理服务
Note over SpotPool : 当价格超过阈值时回收
SpotPool->>Node : 发送驱逐信号
Node->>Pod : 优雅关闭
Pod->>Scheduler : 重新调度到新实例
```

### 成本优化建议

1. **实例类型选择**: 根据模型大小选择合适的GPU实例
2. **Spot实例使用**: 对可容忍中断的工作负载使用Spot实例
3. **自动伸缩**: 配置HPA根据CPU和内存使用率自动调整
4. **存储优化**: 使用Azure Files进行共享存储，避免重复数据
5. **网络优化**: 合理配置负载均衡器和防火墙规则

## 故障排除指南

### 常见问题诊断

#### GPU资源不可用

```mermaid
flowchart TD
A[GPU资源不可用] --> B{检查GPU驱动}
B --> |未安装| C[安装GPU Operator]
B --> |已安装| D{检查节点标签}
D --> |标签缺失| E[添加GPU标签]
D --> |标签正确| F{检查Pod亲和性}
F --> |配置错误| G[修正亲和性规则]
F --> |配置正确| H{检查资源请求}
H --> |请求过高| I[降低资源请求]
H --> |请求正常| J[联系Azure支持]
```

#### 存储挂载失败

```mermaid
sequenceDiagram
participant User as 用户
participant K8s as Kubernetes
participant CSI as CSI驱动
participant Azure as Azure存储
User->>K8s : 创建PVC
K8s->>CSI : 调用挂载
CSI->>Azure : 连接存储
Azure-->>CSI : 返回连接状态
CSI-->>K8s : 挂载结果
K8s-->>User : 显示错误信息
Note over User,K8s : 检查网络和权限
```

#### 网络连接问题

```mermaid
flowchart LR
A[服务不可达] --> B{检查Ingress}
B --> |配置错误| C[修正Ingress规则]
B --> |配置正确| D{检查网络策略}
D --> |策略阻止| E[调整网络策略]
D --> |策略正常| F{检查防火墙}
F --> |防火墙阻断| G[配置防火墙规则]
F --> |防火墙正常| H{检查负载均衡器}
H --> |LB配置错误| I[修正LB配置]
```

### 监控和日志分析

#### 关键指标监控

| 指标类别 | 关键指标 | 阈值设置 | 告警级别 |
|----------|----------|----------|----------|
| GPU性能 | 利用率 > 80% | 80% | 中等 |
| 内存使用 | 使用率 > 85% | 85% | 高 |
| 网络延迟 | P95 > 100ms | 100ms | 中等 |
| 服务可用性 | 可用性 < 99% | 99% | 高 |
| 存储IO | IOPS < 预期值的50% | 50% | 中等 |

#### 日志分析流程

```mermaid
flowchart TD
A[收集日志] --> B{日志分类}
B --> C[应用日志]
B --> D[系统日志]
B --> E[网络日志]
C --> F{分析错误模式}
D --> F
E --> F
F --> G{识别根因}
G --> H{制定修复方案}
H --> I{实施修复}
I --> J{验证修复效果}
```

**章节来源**
- [GPU库存检查工具](file://deploy/utils/gpu_inventory.py#L152-L190)
- [Kubernetes工具集](file://deploy/utils/kubernetes.py#L55-L108)

## 结论

Dynamo在Azure AKS上的部署提供了一个完整的、生产就绪的推理服务平台。通过合理配置GPU节点池、存储系统和网络访问，可以实现高性能、高可用的AI推理服务。

关键成功因素包括：
- 选择合适的GPU VM SKU以平衡性能和成本
- 正确配置存储持久化以确保数据可靠性
- 实施全面的监控和告警机制
- 建立完善的故障排除和恢复流程
- 采用成本优化策略如Spot实例和自动伸缩

这个部署方案为大规模AI推理应用提供了坚实的技术基础，可以根据具体业务需求进行进一步定制和优化。