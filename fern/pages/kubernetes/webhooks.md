---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Webhooks"
---

This document describes the webhook functionality in the Dynamo Operator, including validation webhooks, certificate management, and troubleshooting.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Configuration](#configuration)
  - [Enabling/Disabling Webhooks](#enablingdisabling-webhooks)
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

- ✅ **Enabled by default** - Zero-touch validation out of the box
- ✅ **Shared certificate infrastructure** - All webhook types use the same TLS certificates
- ✅ **Automatic certificate generation** - No manual certificate management required
- ✅ **Defense in depth** - Controllers validate when webhooks are disabled
- ✅ **cert-manager integration** - Optional integration for automated certificate lifecycle
- ✅ **Multi-operator support** - Lease-based coordination for cluster-wide and namespace-restricted deployments
- ✅ **Immutability enforcement** - Critical fields protected via CEL validation rules

### Current Webhook Types

- **Validating Webhooks**: Validate custom resource specifications before persistence
  - `DynamoComponentDeployment` validation
  - `DynamoGraphDeployment` validation
  - `DynamoModel` validation

**Note:** Future releases may add mutating webhooks (for defaults/transformations) and conversion webhooks (for CRD version migrations). All will use the same certificate infrastructure described in this document.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Server                               │
│  1. User submits CR (kubectl apply)                             │
│  2. API server calls ValidatingWebhookConfiguration             │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTPS (TLS required)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Webhook Server (in Operator Pod)                │
│  3. Validates CR against business rules                         │
│  4. Returns admit/deny decision + warnings                      │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Server                                  │
│  5. If admitted: Persist CR to etcd                             │
│  6. If denied: Return error to user                             │
└─────────────────────────────────────────────────────────────────┘
```

### Validation Flow

1. **Webhook validation** (if enabled): Validates at API server level
2. **CEL validation**: Kubernetes-native immutability checks (always active)
3. **Controller validation** (if webhooks disabled): Defense-in-depth validation during reconciliation

---

## Configuration

### Enabling/Disabling Webhooks

Webhooks are **enabled by default**. To disable them:

```yaml
# Platform-level values.yaml
dynamo-operator:
  webhook:
    enabled: false
```

**When to disable webhooks:**
- During development/testing when rapid iteration is needed
- In environments where admission webhooks are not supported
- When troubleshooting validation issues

**Note:** When webhooks are disabled, controllers perform validation during reconciliation (defense in depth).

---

### Certificate Management Options

The operator supports three certificate management modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| **Automatic (Default)** | Helm hooks generate self-signed certificates | Testing and development environments |
| **cert-manager** | Integrate with cert-manager for automated lifecycle | Production deployments with cert-manager |
| **External** | Bring your own certificates | Production deployments with custom PKI |

---

### Advanced Configuration

#### Complete Configuration Reference

```yaml
dynamo-operator:
  webhook:
    # Enable/disable validation webhooks
    enabled: true

    # Certificate management
    certManager:
      enabled: false
      issuerRef:
        kind: Issuer
        name: selfsigned-issuer

    # Certificate secret configuration
    certificateSecret:
      name: webhook-server-cert
      external: false

    # Certificate validity period (automatic generation only)
    certificateValidity: 3650  # 10 years

    # Certificate generator image (automatic generation only)
    certGenerator:
      image:
        repository: bitnami/kubectl
        tag: latest

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

**Zero configuration required!** Certificates are automatically generated during `helm install` and `helm upgrade`.

#### How It Works

1. **Pre-install/pre-upgrade hook**: Generates self-signed TLS certificates
   - Root CA (valid 10 years)
   - Server certificate (valid 10 years)
   - Stores in Secret: `<release>-webhook-server-cert`

2. **Post-install/post-upgrade hook**: Injects CA bundle into `ValidatingWebhookConfiguration`
   - Reads `ca.crt` from Secret
   - Patches `ValidatingWebhookConfiguration` with base64-encoded CA bundle

3. **Operator pod**: Mounts certificate secret and serves webhook on port 9443

#### Certificate Validity

- **Root CA**: 10 years
- **Server Certificate**: 10 years (same as Root CA)
- **Automatic rotation**: Certificates are re-generated on every `helm upgrade`

#### Smart Certificate Generation

The certificate generation hook is intelligent:
- ✅ **Checks existing certificates** before generating new ones
- ✅ **Skips generation** if valid certificates exist (valid for 30+ days with correct SANs)
- ✅ **Regenerates** only when needed (missing, expiring soon, or incorrect SANs)

This means:
- Fast `helm upgrade` operations (no unnecessary cert generation)
- Safe to run `helm upgrade` frequently
- Certificates persist across reinstalls (stored in Secret)

#### Manual Certificate Rotation

If you need to rotate certificates manually:

```bash
# Delete the certificate secret
kubectl delete secret <release>-webhook-server-cert -n <namespace>

# Upgrade the release to regenerate certificates
helm upgrade <release> dynamo-platform -n <namespace>
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

#### Benefits Over Automatic Mode

- ✅ **Automated rotation**: cert-manager renews certificates before expiration
- ✅ **Custom validity periods**: Configure certificate lifetime
- ✅ **CA rotation support**: ca-injector handles CA updates automatically
- ✅ **Integration with existing PKI**: Use your organization's certificate infrastructure

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

1. **Verify webhook is enabled**:
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
# Should see: "Webhooks are enabled - webhooks will validate, controllers will skip validation"
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

4. **Check CA injection job logs**:
```bash
kubectl logs -n <namespace> job/<release>-webhook-ca-inject-<revision>
```

---

### Helm Hook Job Failures

**Symptoms:**
- `helm install` or `helm upgrade` hangs or fails
- Certificate generation errors

**Checks:**

1. **List hook jobs**:
```bash
kubectl get jobs -n <namespace> | grep webhook
```

2. **Check job logs**:
```bash
# Certificate generation
kubectl logs -n <namespace> job/<release>-webhook-cert-gen-<revision>

# CA injection
kubectl logs -n <namespace> job/<release>-webhook-ca-inject-<revision>
```

3. **Check RBAC permissions**:
```bash
# Verify ServiceAccount exists
kubectl get sa -n <namespace> <release>-webhook-ca-inject

# Verify ClusterRole and ClusterRoleBinding exist
kubectl get clusterrole <release>-webhook-ca-inject
kubectl get clusterrolebinding <release>-webhook-ca-inject
```

4. **Manual cleanup**:
```bash
# Delete failed jobs
kubectl delete job -n <namespace> <release>-webhook-cert-gen-<revision>
kubectl delete job -n <namespace> <release>-webhook-ca-inject-<revision>

# Retry helm upgrade
helm upgrade <release> dynamo-platform -n <namespace>
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

2. **Temporarily disable webhook**:
```bash
# Option 1: Delete ValidatingWebhookConfiguration
kubectl delete validatingwebhookconfiguration <name>

# Option 2: Set failurePolicy to Ignore
kubectl patch validatingwebhookconfiguration <name> \
  --type='json' \
  -p='[{"op": "replace", "path": "/webhooks/0/failurePolicy", "value": "Ignore"}]'
```

3. **Delete resource again**:
```bash
kubectl delete <resource-type> <name> -n <namespace>
```

4. **Re-enable webhook**:
```bash
helm upgrade <release> dynamo-platform -n <namespace>
```

---

## Best Practices

### Production Deployments

1. ✅ **Keep webhooks enabled** (default) for real-time validation
2. ✅ **Use `failurePolicy: Fail`** (default) to ensure validation is enforced
3. ✅ **Monitor webhook latency** - Validation adds ~10-50ms per resource operation
4. ✅ **Use cert-manager** for automated certificate lifecycle in large deployments
5. ✅ **Test webhook configuration** in staging before production

### Development Deployments

1. ✅ **Disable webhooks** for rapid iteration if needed
2. ✅ **Use `failurePolicy: Ignore`** if webhook availability is problematic
3. ✅ **Keep automatic certificates** (simpler than cert-manager for dev)

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

