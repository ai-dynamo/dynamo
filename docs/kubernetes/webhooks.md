---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Webhooks
---

This document describes the webhook functionality in the Dynamo Operator, including validation webhooks, certificate management, and troubleshooting.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Configuration](#configuration)
  - [Certificate Management Options](#certificate-management-options)
  - [Advanced Configuration](#advanced-configuration)
- [Certificate Management](#certificate-management)
  - [Automatic Certificates (Default)](#automatic-certificates-default)
  - [cert-manager Integration](#cert-manager-integration)
  - [External Certificates](#external-certificates)
- [Multi-Operator Deployments](#multi-operator-deployments)
- [Troubleshooting](#troubleshooting)

---

## Overview

The Dynamo Operator uses **Kubernetes admission webhooks** to provide real-time validation and mutation of custom resources. Currently, the operator implements **validation webhooks** that ensure invalid configurations are rejected immediately at the API server level, providing faster feedback to users compared to controller-based validation.

All webhook types (validating, mutating, conversion, etc.) share the same **webhook server** and **TLS certificate infrastructure**, making certificate management consistent across all webhook operations.

### Key Features

- ✅ **Always enabled** - Webhooks are a required component of the operator
- ✅ **Shared certificate infrastructure** - All webhook types use the same TLS certificates
- ✅ **Automatic certificate generation and rotation** - Built-in cert-controller, no manual management required
- ✅ **cert-manager integration** - Optional integration for custom PKI or organizational certificate policies
- ✅ **Multi-operator support** - Lease-based coordination for cluster-wide and namespace-restricted deployments
- ✅ **Immutability enforcement** - Critical fields protected via CEL validation rules

### Current Webhook Types

- **Validating Webhooks**: Validate custom resource specifications before persistence
  - `DynamoComponentDeployment` validation
  - `DynamoGraphDeployment` validation
  - `DynamoModel` validation
  - `DynamoGraphDeploymentRequest` validation
- **Mutating Webhooks**: Apply default values to resources on creation
  - `DynamoGraphDeployment` defaulting

**Note:** All webhook types use the same certificate infrastructure described in this document.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Server                               │
│  1. User submits CR (kubectl apply)                             │
│  2. API server calls MutatingWebhookConfiguration               │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTPS (TLS required)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Webhook Server (in Operator Pod)                │
│  3. Applies defaults (e.g., operator version annotation)        │
│  4. Returns mutated CR                                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                         API Server                               │
│  5. API server calls ValidatingWebhookConfiguration             │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTPS (TLS required)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Webhook Server (in Operator Pod)                │
│  6. Validates CR against business rules                         │
│  7. Returns admit/deny decision + warnings                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Server                                  │
│  8. If admitted: Persist CR to etcd                             │
│  9. If denied: Return error to user                             │
└─────────────────────────────────────────────────────────────────┘
```

### Admission Flow

1. **Mutating webhooks**: Apply defaults and transformations before validation
2. **Validating webhooks**: Validate the (possibly mutated) CR against business rules
3. **CEL validation**: Kubernetes-native immutability checks (always active)

---

## Upgrading from versions with `webhook.enabled: false`

The `webhook.enabled` Helm value has been removed. Webhooks are now a required component of the operator and are always active. If you previously ran with `webhook.enabled: false`, take the following steps before upgrading:

1. **Remove `webhook.enabled`** from any custom values files. Helm will ignore the unknown key, but it should be cleaned up to avoid confusion.
2. **Ensure port 9443 is reachable** from the Kubernetes API server to the operator pod. If you have `NetworkPolicy` rules or firewall configurations restricting traffic, add an ingress rule allowing the API server to reach the webhook server on port 9443.
3. **Ensure webhook TLS certificates are available.** By default, the operator's built-in cert-controller generates and rotates self-signed certificates automatically at startup — no action needed. If you use cert-manager or externally managed certificates, verify your configuration is in place before upgrading.

---

## Configuration

### Certificate Management Options

The operator supports three certificate management modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| **Automatic (Default)** | Operator's built-in cert-controller generates and rotates certificates | All environments (recommended) |
| **cert-manager** | Integrate with cert-manager for certificate lifecycle management | Clusters with cert-manager and custom PKI requirements |
| **External** | Bring your own certificates | Environments with externally managed PKI |

---

### Advanced Configuration

#### Complete Configuration Reference

```yaml
dynamo-operator:
  webhook:
    # Certificate management (optional, to use cert-manager instead of built-in)
    certManager:
      enabled: false
      issuerRef:
        kind: Issuer
        name: selfsigned-issuer

    # Certificate secret configuration
    certificateSecret:
      name: webhook-server-cert
      external: false           # Set to true for externally managed certificates

    # Webhook behavior configuration
    failurePolicy: Fail        # Fail (reject on error) or Ignore (allow on error)
    timeoutSeconds: 10         # Webhook timeout

    # Namespace filtering (advanced)
    namespaceSelector: {}      # Kubernetes label selector for namespaces
```

#### Failure Policy

```yaml
# Fail: Reject resources if webhook is unavailable (recommended for production)
webhook:
  failurePolicy: Fail

# Ignore: Allow resources if webhook is unavailable (use with caution)
webhook:
  failurePolicy: Ignore
```

**Recommendation:** Use `Fail` in production to ensure validation is always enforced. Only use `Ignore` if you need high availability and can tolerate occasional invalid resources.

#### Namespace Filtering

Control which namespaces are validated (applies to **cluster-wide operator** only):

```yaml
# Only validate resources in namespaces with specific labels
webhook:
  namespaceSelector:
    matchLabels:
      dynamo-validation: enabled

# Or exclude specific namespaces
webhook:
  namespaceSelector:
    matchExpressions:
    - key: dynamo-validation
      operator: NotIn
      values: ["disabled"]
```

**Note:** For **namespace-restricted operators**, the namespace selector is automatically set to validate only the operator's namespace. This configuration is ignored in namespace-restricted mode.

---

## Certificate Management

### Automatic Certificates (Default)

**Zero configuration required!** The operator's built-in cert-controller generates and rotates certificates automatically at startup.

#### How It Works

1. **Operator starts**: The `CertManager` checks for an existing certificate Secret (configured via `webhook.certificateSecret.name`, default: `webhook-server-cert`). If missing or invalid, it generates a self-signed Root CA and server certificate and writes them to the Secret.

2. **CA bundle injection**: The `CABundleInjector` reads `ca.crt` from the Secret and patches both the `ValidatingWebhookConfiguration` and `MutatingWebhookConfiguration` with the base64-encoded CA bundle.

3. **Certificate rotation**: The cert-controller monitors certificate validity and regenerates certificates before they expire.

4. **Webhook server starts**: The webhook server only begins serving after certificates are confirmed ready, preventing startup races.

#### Certificate Validity

- **Root CA**: 10 years
- **Server Certificate**: 10 years (same as Root CA)
- **Automatic rotation**: The cert-controller monitors validity and regenerates before expiration

#### Smart Certificate Management

The cert-controller is intelligent about certificate lifecycle:
- ✅ **Checks existing certificates** at startup before generating new ones
- ✅ **Skips generation** if valid certificates already exist in the Secret
- ✅ **Regenerates** only when needed (missing, expiring soon, or incorrect SANs)

This means:
- Fast operator restarts (no unnecessary cert generation)
- No dependency on Helm hooks or external Jobs
- Certificates persist across pod restarts (stored in Secret)

#### Manual Certificate Rotation

If you need to rotate certificates manually:

```bash
# Delete the certificate secret -- the operator will regenerate it on restart
kubectl delete secret <release>-webhook-server-cert -n <namespace>

# Restart the operator pod to trigger regeneration
kubectl rollout restart deployment/<release>-dynamo-operator -n <namespace>
```

---

### cert-manager Integration

For clusters with cert-manager installed, you can enable automated certificate lifecycle management.

#### Prerequisites

1. **cert-manager installed** (v1.0+)
2. **CA issuer configured** (e.g., `selfsigned-issuer`)

#### Configuration

```yaml
dynamo-operator:
  webhook:
    certManager:
      enabled: true
      issuerRef:
        kind: Issuer              # Or ClusterIssuer
        name: selfsigned-issuer   # Your issuer name
```

#### How It Works

1. **Helm creates Certificate resource**: Requests TLS certificate from cert-manager
2. **cert-manager generates certificate**: Based on configured issuer
3. **cert-manager stores in Secret**: `<release>-webhook-server-cert`
4. **cert-manager ca-injector**: Automatically injects CA bundle into `ValidatingWebhookConfiguration`
5. **Operator pod**: Mounts certificate secret and serves webhook

#### When to Use cert-manager

- ✅ **Custom validity periods**: Configure certificate lifetime to match organizational policy
- ✅ **Integration with existing PKI**: Use your organization's certificate infrastructure
- ✅ **Centralized certificate management**: Manage all cluster certificates through cert-manager

#### Certificate Rotation

With cert-manager, certificate rotation is **fully automated**:

1. **Leaf certificate rotation** (default: every year)
   - cert-manager auto-renews before expiration
   - controller-runtime auto-reloads new certificate
   - **No pod restart required**
   - **No caBundle update required** (same Root CA)

2. **Root CA rotation** (every 10 years)
   - cert-manager rotates Root CA
   - ca-injector auto-updates caBundle in `ValidatingWebhookConfiguration`
   - **No manual intervention required**

#### Example: Self-Signed Issuer

```yaml
apiVersion: cert-manager.io/v1
kind: Issuer
metadata:
  name: selfsigned-issuer
  namespace: dynamo-system
spec:
  selfSigned: {}
---
# Enable in platform values.yaml
dynamo-operator:
  webhook:
    certManager:
      enabled: true
      issuerRef:
        kind: Issuer
        name: selfsigned-issuer
```

---

### External Certificates

Bring your own certificates for custom PKI requirements.

#### Steps

1. **Create certificate secret manually**:

```bash
kubectl create secret tls <release>-webhook-server-cert \
  --cert=tls.crt \
  --key=tls.key \
  -n <namespace>

# Also add ca.crt to the secret
kubectl patch secret <release>-webhook-server-cert -n <namespace> \
  --type='json' \
  -p='[{"op": "add", "path": "/data/ca.crt", "value": "'$(base64 -w0 < ca.crt)'"}]'
```

2. **Configure operator to use external secret**:

```yaml
dynamo-operator:
  webhook:
    certificateSecret:
      external: true
    caBundle: <base64-encoded-ca-cert>  # Must manually specify
```

3. **Deploy operator**:

```bash
helm install dynamo-platform . -n <namespace> -f values.yaml
```

#### Certificate Requirements

- **Secret name**: Must match `webhook.certificateSecret.name` (default: `webhook-server-cert`)
- **Secret keys**: `tls.crt`, `tls.key`, `ca.crt`
- **Certificate SAN**: Must include `<service-name>.<namespace>.svc`
  - Example: `dynamo-platform-dynamo-operator-webhook-service.dynamo-system.svc`

---

## Multi-Operator Deployments

The operator supports running both **cluster-wide** and **namespace-restricted** instances simultaneously using a **lease-based coordination mechanism**.

### Scenario

```
Cluster:
├─ Operator A (cluster-wide, namespace: platform-system)
│  └─ Validates all namespaces EXCEPT team-a
└─ Operator B (namespace-restricted, namespace: team-a)
   └─ Validates only team-a namespace
```

### How It Works

1. **Namespace-restricted operator** creates a Lease in its namespace
2. **Cluster-wide operator** watches for Leases named `dynamo-operator-ns-lock`
3. **Cluster-wide operator** skips validation for namespaces with active Leases
4. **Namespace-restricted operator** validates resources in its namespace

### Lease Configuration

The lease mechanism is **automatically configured** based on deployment mode:

```yaml
# Cluster-wide operator (default)
namespaceRestriction:
  enabled: false
# → Watches for leases in all namespaces
# → Skips validation for namespaces with active leases

# Namespace-restricted operator
namespaceRestriction:
  enabled: true
  namespace: team-a
# → Creates lease in team-a namespace
# → Does NOT check for leases (no cluster permissions)
```

### Deployment Example

```bash
# 1. Deploy cluster-wide operator
helm install platform-operator dynamo-platform \
  -n platform-system \
  --set namespaceRestriction.enabled=false

# 2. Deploy namespace-restricted operator for team-a
helm install team-a-operator dynamo-platform \
  -n team-a \
  --set namespaceRestriction.enabled=true \
  --set namespaceRestriction.namespace=team-a
```

### ValidatingWebhookConfiguration Naming

The webhook configuration name reflects the deployment mode:

- **Cluster-wide**: `<release>-validating`
- **Namespace-restricted**: `<release>-validating-<namespace>`

Example:

```bash
# Cluster-wide
platform-operator-validating

# Namespace-restricted (team-a)
team-a-operator-validating-team-a
```

This allows multiple webhook configurations to coexist without conflicts.

### Lease Health

If the namespace-restricted operator is deleted or becomes unhealthy:
- Lease expires after `leaseDuration + gracePeriod` (default: ~30 seconds)
- Cluster-wide operator automatically resumes validation for that namespace

---

## Troubleshooting

### Webhook Not Called

**Symptoms:**
- Invalid resources are accepted
- No validation errors in logs

**Checks:**

1. **Verify webhook configuration exists**:
```bash
kubectl get validatingwebhookconfiguration | grep dynamo
```

2. **Check webhook configuration**:
```bash
kubectl get validatingwebhookconfiguration <name> -o yaml
# Verify:
# - caBundle is present and non-empty
# - clientConfig.service points to correct service
# - webhooks[].namespaceSelector matches your namespace
```

3. **Verify webhook service exists**:
```bash
kubectl get service -n <namespace> | grep webhook
```

4. **Check operator logs for webhook startup**:
```bash
kubectl logs -n <namespace> deployment/<release>-dynamo-operator | grep webhook
# Should see: "Registering validation webhooks"
# Should see: "Starting webhook server"
```

---

### Connection Refused Errors

**Symptoms:**
```
Error from server (InternalError): Internal error occurred: failed calling webhook:
Post "https://...webhook-service...:443/validate-...": dial tcp ...:443: connect: connection refused
```

**Checks:**

1. **Verify operator pod is running**:
```bash
kubectl get pods -n <namespace> -l app.kubernetes.io/name=dynamo-operator
```

2. **Check webhook server is listening**:
```bash
# Port-forward to pod
kubectl port-forward -n <namespace> pod/<operator-pod> 9443:9443

# In another terminal, test connection
curl -k https://localhost:9443/validate-nvidia-com-v1alpha1-dynamocomponentdeployment
# Should NOT get "connection refused"
```

3. **Verify webhook port in deployment**:
```bash
kubectl get deployment -n <namespace> <release>-dynamo-operator -o yaml | grep -A5 "containerPort: 9443"
```

4. **Check for webhook initialization errors**:
```bash
kubectl logs -n <namespace> deployment/<release>-dynamo-operator | grep -i error
```

---

### Certificate Errors

**Symptoms:**
```
Error from server (InternalError): Internal error occurred: failed calling webhook:
x509: certificate signed by unknown authority
```

**Checks:**

1. **Verify caBundle is present**:
```bash
kubectl get validatingwebhookconfiguration <name> -o jsonpath='{.webhooks[0].clientConfig.caBundle}' | base64 -d
# Should output a valid PEM certificate
```

2. **Verify certificate secret exists**:
```bash
kubectl get secret -n <namespace> <release>-webhook-server-cert
```

3. **Check certificate validity**:
```bash
kubectl get secret -n <namespace> <release>-webhook-server-cert -o jsonpath='{.data.tls\.crt}' | base64 -d | openssl x509 -noout -text
# Check:
# - Not expired
# - SAN includes: <service-name>.<namespace>.svc
```

4. **Check operator logs for CA injection errors**:
```bash
kubectl logs -n <namespace> deployment/<release>-dynamo-operator | grep -i "cert\|ca.*bundle\|inject"
```

---

### Certificate Controller Errors

**Symptoms:**
- Operator logs show cert-controller errors
- Certificate Secret is not created
- CA bundle is not injected into webhook configurations

**Checks:**

1. **Check cert-controller logs**:
```bash
kubectl logs -n <namespace> deployment/<release>-dynamo-operator | grep -i "cert-manager\|cert-rotation\|cert-controller"
```

2. **Verify RBAC permissions**:
```bash
# The operator needs permissions to manage Secrets, ValidatingWebhookConfigurations,
# MutatingWebhookConfigurations, and CustomResourceDefinitions
kubectl auth can-i create secrets -n <namespace> --as=system:serviceaccount:<namespace>:<release>-dynamo-operator
kubectl auth can-i patch validatingwebhookconfigurations --as=system:serviceaccount:<namespace>:<release>-dynamo-operator
```

3. **Check if the certificate Secret was created**:
```bash
kubectl get secret -n <namespace> <release>-webhook-server-cert
```

4. **Force certificate regeneration**:
```bash
# Delete the certificate secret and restart the operator
kubectl delete secret <release>-webhook-server-cert -n <namespace>
kubectl rollout restart deployment/<release>-dynamo-operator -n <namespace>
```

---

### Validation Errors Not Clear

**Symptoms:**
- Webhook rejects resource but error message is unclear

**Solution:**

Check operator logs for detailed validation errors:

```bash
kubectl logs -n <namespace> deployment/<release>-dynamo-operator | grep "validate create\|validate update"
```

Webhook logs include:
- Resource name and namespace
- Validation errors with context
- Warnings for immutable field changes

---

### Stuck Deleting Resources

**Symptoms:**
- Resource stuck in "Terminating" state
- Webhook blocks finalizer removal

**Solution:**

The webhook automatically skips validation for resources being deleted. If stuck:

1. **Check if webhook is blocking**:
```bash
kubectl describe <resource-type> <name> -n <namespace>
# Look for events mentioning webhook errors
```

2. **Temporarily work around the webhook**:
```bash
# Option 1: Set failurePolicy to Ignore
kubectl patch validatingwebhookconfiguration <name> \
  --type='json' \
  -p='[{"op": "replace", "path": "/webhooks/0/failurePolicy", "value": "Ignore"}]'

# Option 2 (last resort): Delete ValidatingWebhookConfiguration
kubectl delete validatingwebhookconfiguration <name>
```

3. **Delete resource again**:
```bash
kubectl delete <resource-type> <name> -n <namespace>
```

4. **Restore webhook configuration**:
```bash
helm upgrade <release> dynamo-platform -n <namespace>
```

---

## Best Practices

### Production Deployments

1. ✅ **Use `failurePolicy: Fail`** (default) to ensure validation is enforced
2. ✅ **Monitor webhook latency** - Validation adds ~10-50ms per resource operation
3. ✅ **Automatic certificates work well for production** - The built-in cert-controller handles generation and rotation; use cert-manager only if you need integration with organizational PKI
4. ✅ **Test webhook configuration** in staging before production

### Development Deployments

1. ✅ **Use `failurePolicy: Ignore`** if webhook availability is problematic during development
2. ✅ **Keep automatic certificates** (zero configuration, built into the operator)

### Multi-Tenant Deployments

1. ✅ **Deploy one cluster-wide operator** for platform-wide validation
2. ✅ **Deploy namespace-restricted operators** for tenant-specific namespaces
3. ✅ **Monitor lease health** to ensure coordination works correctly
4. ✅ **Use unique release names** per namespace to avoid naming conflicts

---

## Additional Resources

- [Kubernetes Admission Webhooks](https://kubernetes.io/docs/reference/access-authn-authz/extensible-admission-controllers/)
- [cert-manager Documentation](https://cert-manager.io/docs/)
- [Kubebuilder Webhook Tutorial](https://book.kubebuilder.io/cronjob-tutorial/webhook-implementation.html)
- [CEL Validation Rules](https://kubernetes.io/docs/reference/using-api/cel/)

---

## Support

For issues or questions:
- Check [Troubleshooting](#troubleshooting) section
- Review operator logs: `kubectl logs -n <namespace> deployment/<release>-dynamo-operator`
- Open an issue on GitHub

