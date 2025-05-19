# Dynamo Kubernetes Operator

A Kubernetes Operator to manage all Dynamo pipelines using custom resources.


## Overview

This operator automates the deployment and lifecycle management of `DynamoGraphDeployment` resources in Kubernetes clusters.

Built with [Kubebuilder](https://book.kubebuilder.io/), it follows Kubernetes best practices and supports declarative configuration through CustomResourceDefinitions (CRDs).

## Developer guide

### Pre-requisites

- [Go](https://go.dev/doc/install) >= 1.24
- [Kubebuilder](https://book.kubebuilder.io/quick-start.html)

### Build

```
make
```
