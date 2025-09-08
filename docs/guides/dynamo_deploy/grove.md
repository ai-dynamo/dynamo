# Grove Deployment Guide

Grove is a Kubernetes API specifically designed to address the orchestration challenges of modern AI workloads, particularly disaggregated inference systems. Grove provides seamless integration with NVIDIA Dynamo for comprehensive AI infrastructure management.

## Overview

Grove was originally motivated by the challenges of orchestrating multinode, disaggregated inference systems. It provides a consistent and unified API that allows users to define, configure, and scale prefill, decode, and any other components like routing within a single custom resource.

### How Grove Works for Disaggregated Serving

Grove enables disaggregated serving by breaking down large language model inference into separate, specialized components that can be independently scaled and managed. This architecture provides several advantages:

- **Component Specialization**: Separate prefill, decode, and routing components optimized for their specific tasks
- **Independent Scaling**: Each component can scale based on its individual resource requirements and workload patterns
- **Resource Optimization**: Better utilization of hardware resources through specialized workload placement
- **Fault Isolation**: Issues in one component don't necessarily affect others

## Core Components and API Resources

Grove implements disaggregated serving through several custom Kubernetes resources that provide declarative composition of role-based pod groups:

### PodGangSet
The top-level Grove object that defines a group of components managed and colocated together. Key features include:
- Support for autoscaling
- Topology-aware spread of replicas for availability
- Unified management of multiple disaggregated components

### PodClique
Represents a group of pods with a specific role (e.g., leader, worker, frontend). Each clique features:
- Independent configuration options
- Custom scaling logic support
- Role-specific resource allocation

### PodCliqueScalingGroup
A set of PodCliques that scale and are scheduled together, ideal for tightly coupled roles like prefill leader and worker components that need coordinated scaling behavior.

## Key Capabilities for Disaggregated Serving

Grove provides several specialized features that make it particularly well-suited for disaggregated serving:

### Flexible Gang Scheduling
PodCliques and PodCliqueScalingGroups allow users to specify flexible gang-scheduling requirements at multiple levels within a PodGangSet to prevent resource deadlocks and ensure all components of a disaggregated system start together.

### Multi-level Horizontal Auto-Scaling
Supports pluggable horizontal auto-scaling solutions to scale PodGangSet, PodClique, and PodCliqueScalingGroup custom resources independently based on their specific metrics and requirements.

### Network Topology-Aware Scheduling
Allows specifying network topology pack and spread constraints to optimize for both network performance and service availability, crucial for disaggregated systems where components need efficient inter-node communication.

### Custom Startup Dependencies
Prescribes the order in which PodCliques must start in a declarative specification, with pod startup decoupled from pod creation or scheduling. This ensures proper initialization order for disaggregated components.

## Use Cases and Examples

Grove specifically supports:

- **Multi-node disaggregated inference** for large models such as DeepSeek-R1 and Llama-4-Maverick
- **Single-node disaggregated inference** for optimized resource utilization
- **Agentic pipelines of models** for complex AI workflows
- **Standard aggregated serving** patterns for single node or single GPU inference

## Integration with NVIDIA Dynamo

Grove is strategically aligned with NVIDIA Dynamo for seamless integration within the AI infrastructure stack:

### Complementary Roles
- **Grove**: Handles the Kubernetes orchestration layer for disaggregated AI workloads
- **Dynamo**: Provides comprehensive AI infrastructure capabilities including serving backends, routing, and resource management

### Release Coordination
Grove is aligning its release schedule with NVIDIA Dynamo to ensure seamless integration, with the finalized release cadence reflected in the project roadmap.

### Unified AI Platform
The integration creates a comprehensive platform where:
- Grove manages complex orchestration of disaggregated components
- Dynamo provides the serving infrastructure, routing capabilities, and backend integrations
- Together they enable sophisticated AI serving architectures with simplified management

## Architecture Benefits

Grove represents a significant advancement in Kubernetes-based orchestration for AI workloads by:

1. **Simplifying Complex Deployments**: Provides a unified API that can manage multiple components (prefill, decode, routing) within a single resource definition
2. **Enabling Sophisticated Architectures**: Supports advanced disaggregated inference patterns that were previously difficult to orchestrate
3. **Reducing Operational Complexity**: Abstracts away the complexity of coordinating multiple interdependent AI components
4. **Optimizing Resource Utilization**: Enables fine-grained control over component placement and scaling

## Getting Started

Grove relies on KAI Scheduler for resource allocation and scheduling.

For KAI Scheduler, see the [KAI Scheduler Deployment Guide](https://github.com/NVIDIA/KAI-Scheduler).

For installation instructions, see the [Grove Installation Guide](https://github.com/NVIDIA/grove/blob/main/docs/installation.md).

For practical examples of Grove-based multinode deployments in action, see the [Multinode Deployment Guide](multinode-deployment.md), which demonstrates multi-node disaggregated serving scenarios.

For the latest updates on Grove, refer to the [official project on GitHub](https://github.com/NVIDIA/grove).

Dynamo Cloud also allows you to install Grove and KAI Scheduler as part of the platform installation. See the [Dynamo Cloud Deployment Installation Guide](installation_guide.md) for more details.