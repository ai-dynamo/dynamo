# Dynamo Deployment Guide

This directory contains all the necessary files and instructions for deploying Dynamo in various environments. Choose the deployment method that best suits your needs:

## Directory Structure

```
deploy/
├── cloud/                    # Cloud deployment configurations and tools
├── helm/                     # Helm charts for manual Kubernetes deployment
├── metrics/                  # Monitoring and metrics configuration
│   ├── docker-compose.yml    # Docker compose for Prometheus and Grafana
│   ├── prometheus.yml        # Prometheus configuration
│   └── README.md             # Metrics setup instructions
├── sdk/                      # Dynamo SDK and related tools
└── README.md                 # This file
```

## Deployment Options

### 1. 🚀 Dynamo Cloud Platform [PREFERRED]

The Dynamo Cloud Platform provides a managed deployment experience with:
- Automated infrastructure management
- Built-in monitoring and metrics
- Simplified deployment process
- Production-ready configurations

For detailed instructions, see:
- [Dynamo Cloud Platform Guide](../docs/guides/dynamo_deploy/dynamo_cloud.md)
- [Operator Deployment Guide](../docs/guides/dynamo_deploy/operator_deployment.md)

### 2. Manual Deployment with Helm Charts

For users who need more control over their deployments:
- Full control over deployment parameters
- Manual management of infrastructure
- Customizable monitoring setup
- Flexible configuration options

Documentation:
- [Manual Helm Deployment Guide](../docs/guides/dynamo_deploy/manual_helm_deployment.md)
- [Minikube Setup Guide](../docs/guides/dynamo_deploy/minikube.md)

## Choosing the Right Deployment Method

- **Dynamo Cloud Platform**: Best for most users, provides managed deployment with built-in monitoring
  - See [Dynamo Cloud Platform Guide](../docs/guides/dynamo_deploy/dynamo_cloud.md)
- **Manual Helm Deployment**: For users who need full control over their deployment
  - See [Manual Helm Deployment Guide](../docs/guides/dynamo_deploy/manual_helm_deployment.md)
