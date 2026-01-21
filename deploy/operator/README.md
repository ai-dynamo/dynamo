# Dynamo Kubernetes Operator

A Kubernetes Operator to manage all Dynamo pipelines using custom resources.


## Overview

This operator automates the deployment and lifecycle management of Dynamo resources in Kubernetes clusters:

- **DynamoGraphDeploymentRequest (DGDR)** - Simplified SLA-driven deployment interface
- **DynamoGraphDeployment (DGD)** - Direct deployment configuration

Built with [Kubebuilder](https://book.kubebuilder.io/), it follows Kubernetes best practices and supports declarative configuration through CustomResourceDefinitions (CRDs).

### Custom Resources

- **DynamoGraphDeploymentRequest**: High-level interface for SLA-driven configuration generation. Automatically handles profiling and generates an optimized DGD spec based on your performance requirements.
- **DynamoGraphDeployment**: Lower-level interface for direct deployment configuration with full control over all parameters.


## Developer guide

### Pre-requisites

- [Go](https://go.dev/doc/install) >= 1.25
- [Kubebuilder](https://book.kubebuilder.io/quick-start.html)

### Build

```
make
```

### Install

See [Dynamo Kubernetes Platform Installation Guide](/docs/kubernetes/installation_guide.md) for installation instructions.
