# Kubernetes Discovery Integration Tests

This directory contains integration tests for the Dynamo Kubernetes discovery client. These tests verify that our Rust code can correctly interact with the Kubernetes API to list and watch EndpointSlices.

## Prerequisites

1. **Kubernetes Cluster Access**: You need a running Kubernetes cluster with `kubectl` configured
   - Local: Docker Desktop, Minikube, Kind, k3s, etc.
   - Cloud: GKE, EKS, AKS, etc.

2. **Admin/Sufficient Permissions**: Your current kubectl context should have permissions to:
   - Create/delete Deployments and Services
   - List/watch EndpointSlices

3. **Rust Environment**: Cargo with the dynamo-runtime crate compiled

## Quick Start

### 1. Deploy Test Resources

```bash
cd k8s-test

# Deploy to default namespace
./deploy.sh

# Or deploy to a specific namespace
./deploy.sh my-namespace
```

This will:
- Create the namespace if it doesn't exist
- Create a deployment with 3 nginx pods
- Create a service that generates EndpointSlices
- Wait for pods to be ready
- Show the current status

**Examples:**
```bash
./deploy.sh                    # Deploy to 'default' namespace
./deploy.sh test-namespace     # Deploy to 'test-namespace'
./deploy.sh production         # Deploy to 'production' namespace
```

### 2. Run Integration Tests

There are two test suites:

#### A. Raw Kubernetes API Tests (kube_discovery_integration)

These tests verify the raw Kubernetes API interactions work correctly:

```bash
# Run all raw K8s tests
cargo test --test kube_discovery_integration -- --ignored --nocapture

# Or run individual tests:
cargo test --test kube_discovery_integration test_kube_client_connection -- --ignored --nocapture
cargo test --test kube_discovery_integration test_list_endpointslices -- --ignored --nocapture
cargo test --test kube_discovery_integration test_watch_endpointslices -- --ignored --nocapture
cargo test --test kube_discovery_integration test_discovery_simulation -- --ignored --nocapture
```

#### B. KubeDiscoveryClient Tests (kube_client_integration) **[RECOMMENDED]**

These tests verify the actual `KubeDiscoveryClient` implementation:

```bash
# Run all KubeDiscoveryClient tests (sequential for clean output)
cargo test --test kube_client_integration -- --ignored --nocapture --test-threads=1

# Or run individual tests:

# Test client creation
cargo test --test kube_client_integration test_client_creation -- --ignored --nocapture

# Test list() method
cargo test --test kube_client_integration test_list_all_endpoints -- --ignored --nocapture
cargo test --test kube_client_integration test_list_namespaced_endpoints -- --ignored --nocapture
cargo test --test kube_client_integration test_list_component_endpoints -- --ignored --nocapture

# Test list_and_watch() method
cargo test --test kube_client_integration test_watch_all_endpoints -- --ignored --nocapture
cargo test --test kube_client_integration test_watch_namespaced_endpoints -- --ignored --nocapture
cargo test --test kube_client_integration test_watch_receives_k8s_events -- --ignored --nocapture
```

**Note:** The `--test-threads=1` flag ensures tests run sequentially, preventing output from multiple tests from being interleaved. This makes the output much more readable, especially for watch tests that print events over time.

**Note:** The `KubeDiscoveryClient` tests use **mock metadata** mode, which means they skip actual HTTP calls to pods and return mock `DiscoveryMetadata` instead. This allows the tests to verify:
- ✅ Kubernetes API interactions (listing/watching EndpointSlices)
- ✅ Endpoint extraction from EndpointSlices
- ✅ Discovery event flow (Added/Removed events)
- ✅ The full discovery pipeline

Without needing actual metadata servers running in pods. This makes tests fast, reliable, and easy to run.

#### Alternative: Using the Test Runner Script

You can also use the `run-tests.sh` script for a more convenient workflow:

```bash
cd k8s-test

# Run all client tests (checks default namespace)
./run-tests.sh

# Run specific test
./run-tests.sh client test_list_all_endpoints

# Run tests and check a specific namespace
./run-tests.sh client "" my-namespace

# Run all test suites
./run-tests.sh all
```

The script will:
- Check if kubectl is configured
- Verify test resources exist in the specified namespace
- Run the requested tests
- Provide helpful error messages if resources aren't deployed

### 3. Clean Up

```bash
# Clean up from default namespace
./cleanup.sh

# Or clean up from a specific namespace
./cleanup.sh my-namespace
```

**Examples:**
```bash
./cleanup.sh                    # Clean up from 'default' namespace
./cleanup.sh test-namespace     # Clean up from 'test-namespace'
./cleanup.sh production         # Clean up from 'production' namespace
```

**Note:** The cleanup script does not delete the namespace itself. To delete the namespace:
```bash
kubectl delete namespace my-namespace
```

## Test Descriptions

### KubeDiscoveryClient Tests (Recommended)

These tests exercise the actual `KubeDiscoveryClient` methods that will be used in production.

#### `test_client_creation`
Verifies that we can create a `KubeDiscoveryClient` for testing.

**What it tests:**
- Client instantiation
- Instance ID generation from pod name

**Expected output:**
```
🔌 Testing KubeDiscoveryClient creation...
✅ Client created with instance_id: abc123def456
```

#### `test_list_all_endpoints`
Tests the `list()` method with `DiscoveryKey::AllEndpoints`.

**What it tests:**
- Calling `KubeDiscoveryClient::list()`
- EndpointSlice querying without label filters
- Metadata fetching workflow (will fail gracefully without metadata server)

**Expected output:**
```
📋 Testing list all endpoints...
   Note: This will try to fetch metadata from pods via HTTP,
   which will likely fail unless pods are running the metadata server.
   The test verifies the Kubernetes API calls work correctly.
Calling list() with key=AllEndpoints
✅ list() succeeded
   Found 0 instances
✅ List test completed (K8s API calls work)
```

#### `test_list_namespaced_endpoints` & `test_list_component_endpoints`
Test the `list()` method with label-based filtering.

**What it tests:**
- Label selector generation from `DiscoveryKey`
- Filtered EndpointSlice queries

#### `test_watch_all_endpoints` & `test_watch_namespaced_endpoints`
Test the `list_and_watch()` method which creates a streaming watch.

**What it tests:**
- Creating a watch stream from `KubeDiscoveryClient`
- Receiving discovery events (Added/Removed)
- Watch lifecycle management

**Expected output:**
```
👀 Testing watch all endpoints...
   This test will watch for 5 seconds
Calling list_and_watch() with key=AllEndpoints
📡 Watch stream started...
⏰ Timeout reached
✅ Watch test completed (0 events received)
   Note: Events are only emitted when pods are discovered
   and their metadata can be fetched via HTTP
```

#### `test_watch_receives_k8s_events`
Verifies the Kubernetes watcher integration is functioning.

**What it tests:**
- Watch stream receives at least one event
- K8s watcher initialization
- Stream lifecycle

### Raw Kubernetes API Tests

These tests verify low-level Kubernetes API interactions.

#### `test_kube_client_connection`
Verifies that we can create a Kubernetes client and connect to the cluster.

**What it tests:**
- Client creation from default kubeconfig
- Basic API connectivity by listing namespaces

**Expected output:**
```
🔌 Testing Kubernetes client connection...
✅ Successfully connected to Kubernetes cluster
📋 Found X namespaces
✅ Kubernetes API is accessible
```

### `test_list_endpointslices`
Tests listing all EndpointSlices in the default namespace.

**What it tests:**
- EndpointSlice API access
- Parsing EndpointSlice structures
- Extracting endpoint information (pod names, IPs, readiness)

**Expected output:**
```
📋 Testing EndpointSlice listing...
📊 Found X EndpointSlices in default namespace
  • dynamo-test-service-abcde (service: dynamo-test-service, endpoints: 3)
    [0] pod=dynamo-test-worker-xxx, ready=true, addresses=["10.1.2.3"]
    [1] pod=dynamo-test-worker-yyy, ready=true, addresses=["10.1.2.4"]
    [2] pod=dynamo-test-worker-zzz, ready=true, addresses=["10.1.2.5"]
✅ EndpointSlice listing test completed
```

### `test_list_with_labels`
Tests listing EndpointSlices with label selectors (like our discovery client does).

**What it tests:**
- Label selector functionality
- Filtering EndpointSlices by labels

**Important:** EndpointSlices are created by Services, not Deployments. The EndpointSlices will have labels from the Service, not from the pod labels. The test uses `kubernetes.io/service-name=dynamo-test-service` which is automatically added by Kubernetes.

**Expected output:**
```
🏷️  Testing EndpointSlice listing with label selector...
Using label selector: kubernetes.io/service-name=dynamo-test-service
📊 Found X EndpointSlices matching labels
  • dynamo-test-service-abcde (endpoints: 3)
✅ Label selector test completed
```

### `test_watch_endpointslices`
Tests the Kubernetes watch mechanism for EndpointSlices.

**What it tests:**
- Creating a watch stream
- Receiving watch events (Init, InitApply, Apply, Delete, InitDone)
- Event types and their contents

**Expected output:**
```
👀 Testing EndpointSlice watching...
   This test will watch for 10 seconds or 5 events, whichever comes first
📡 Watch stream started...
  [1] 🚀 Init - watch stream starting
  [2] 🔄 InitApply: dynamo-test-service-xxx (endpoints: 3)
  [3] ✅ InitDone - initial list complete
📊 Reached max events (5), stopping watch
✅ Watch test completed (5 events received)
```

### `test_watch_with_labels`
Tests watching EndpointSlices with a label selector.

**What it tests:**
- Watch with label filtering
- Receiving only relevant events

**Expected output:**
```
👀 Testing EndpointSlice watching with label selector...
   This test will watch for 5 seconds or until InitDone
Using label selector: kubernetes.io/service-name=dynamo-test-service
📡 Watch stream started...
  [1] 🚀 Init - watch stream starting
  [2] 🔄 InitApply: dynamo-test-service-xxx (endpoints: 3)
  [3] ✅ InitDone - initial list complete
📊 InitDone received, stopping watch
✅ Watch with labels test completed (3 events received)
```

### `test_discovery_simulation`
Comprehensive test that simulates the full discovery client behavior.

**What it tests:**
- Complete discovery flow: watch → extract endpoints → track instances
- Pod name hashing (instance ID generation)
- Ready state filtering
- Duplicate detection

**Expected output:**
```
🔍 Testing discovery client simulation...
   This simulates how our KubeDiscoveryClient list_and_watch works
Label selector: kubernetes.io/service-name=dynamo-test-service
📡 Starting watch stream...
  🚀 Watch stream initialized
  📦 Processing EndpointSlice: dynamo-test-service-xxx
    ✅ New endpoint: pod=dynamo-test-worker-xxx, instance_id=abc123, addresses=["10.1.2.3"]
    ✅ New endpoint: pod=dynamo-test-worker-yyy, instance_id=def456, addresses=["10.1.2.4"]
    ✅ New endpoint: pod=dynamo-test-worker-zzz, instance_id=789abc, addresses=["10.1.2.5"]
  ✅ Initial sync complete
  📊 Discovered 3 unique endpoints
✅ Discovery simulation completed
📊 Total unique endpoints discovered: 3
```

## Architecture Overview

```
┌─────────────────────────────────────────┐
│         Kubernetes Cluster              │
│                                         │
│  Namespace: default (POD_NAMESPACE)    │
│  ┌─────────────────────────────────┐   │
│  │  Deployment: dynamo-test-worker │   │
│  │  Replicas: 3                    │   │
│  │  Labels:                        │   │
│  │    app=dynamo-test              │   │
│  │    component=worker             │   │
│  └─────────────────────────────────┘   │
│            │                            │
│            ▼                            │
│  ┌─────────────────────────────────┐   │
│  │  Pods (3 replicas)              │   │
│  │  - dynamo-test-worker-xxx       │   │
│  │  - dynamo-test-worker-yyy       │   │
│  │  - dynamo-test-worker-zzz       │   │
│  └─────────────────────────────────┘   │
│            │                            │
│            ▼                            │
│  ┌─────────────────────────────────┐   │
│  │  Service: dynamo-test-service   │   │
│  │  Type: ClusterIP                │   │
│  │  Selector: app=dynamo-test      │   │
│  └─────────────────────────────────┘   │
│            │                            │
│            ▼                            │
│  ┌─────────────────────────────────┐   │
│  │  EndpointSlices (auto-created)  │   │
│  │  Labels:                        │   │
│  │    kubernetes.io/service-name:  │   │
│  │      dynamo-test-service        │   │
│  │  Endpoints: [pod IPs + status]  │   │
│  └─────────────────────────────────┘   │
│                                         │
└─────────────────────────────────────────┘
                 │
                 │ Kubernetes API
                 │ (List/Watch - namespace scoped)
                 ▼
┌─────────────────────────────────────────┐
│     Integration Tests (Rust)            │
│  - test_list_endpointslices             │
│  - test_watch_endpointslices            │
│  - test_discovery_simulation            │
└─────────────────────────────────────────┘
```

**Important:** The `KubeDiscoveryClient` is **namespace-scoped**. It only watches EndpointSlices in the namespace specified by the `POD_NAMESPACE` environment variable. This provides:
- ✅ Better security (no cluster-wide access needed)
- ✅ Better performance (fewer resources to watch)
- ✅ Namespace isolation (pods only discover within their namespace)

## Troubleshooting

### "Failed to create Kubernetes client"

**Cause:** kubectl is not configured or kubeconfig is invalid

**Solution:**
```bash
# Check kubectl connection
kubectl cluster-info

# Check current context
kubectl config current-context

# If needed, set context
kubectl config use-context <context-name>
```

### "No EndpointSlices found"

**Cause:** Test resources not deployed

**Solution:**
```bash
cd k8s-test
./deploy.sh

# Verify resources exist
kubectl get endpointslices -l kubernetes.io/service-name=dynamo-test-service
```

### "Pods not ready"

**Cause:** Pods are still starting or failing

**Solution:**
```bash
# Check pod status
kubectl get pods -l app=dynamo-test

# Check pod events
kubectl describe pod <pod-name>

# Check pod logs
kubectl logs <pod-name>
```

### "No endpoints discovered"

**Cause:** Pods might not be ready yet

**Solution:**
```bash
# Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app=dynamo-test --timeout=60s

# Check pod readiness
kubectl get pods -l app=dynamo-test -o wide
```

## Notes

### Namespace Configuration

The `KubeDiscoveryClient` reads the `POD_NAMESPACE` environment variable to determine which namespace to watch. This is automatically set by Kubernetes when you use the downward API:

```yaml
env:
- name: POD_NAMESPACE
  valueFrom:
    fieldRef:
      fieldPath: metadata.namespace
```

The client will **only** watch EndpointSlices within this namespace. It does not have cluster-wide access.

### Why EndpointSlices?

Kubernetes automatically creates EndpointSlices for Services. EndpointSlices track:
- Pod IPs
- Pod readiness state
- Pod names (via targetRef)
- Port information

This makes them perfect for service discovery.

### Labels on EndpointSlices

**Important:** EndpointSlices inherit labels from the Service, not from Pods. The most reliable label to use is:
- `kubernetes.io/service-name=<service-name>` (automatically added)

If you want custom labels on EndpointSlices, add them to the Service, not the Pods.

### Difference from Production

These integration tests skip the HTTP metadata fetching part. In production:
1. Watch EndpointSlice → get pod IPs
2. HTTP GET `http://<pod-ip>:8080/metadata` → get registration data
3. Cache and return discovery instances

For these tests, we only verify step 1 works correctly.

## Local Testing Mode

Want to test with a **real metadata server** running locally? See **[LOCAL_TESTING.md](LOCAL_TESTING.md)** for detailed instructions.

Quick start:
```bash
# 1. Create a test pod and service with custom labels
./create-local-test-pod.sh 9000 discovery hello_world backend
#                          ^port ^k8s-ns  ^dynamo-ns  ^component

# This creates:
# - Pod: dynamo-test-worker-9000
# - Service: dynamo-test-service-9000
# - EndpointSlice: (auto-created by K8s)

# 2. Start your metadata server locally (in another terminal)
export PORT=9000
export DYN_SYSTEM_PORT=$PORT
export POD_NAME=dynamo-test-worker-$PORT
export POD_NAMESPACE=discovery
export DYN_DISCOVERY_BACKEND=kubernetes
python3 -m your_app

# 3. Run your client in local mode
export DYN_LOCAL_KUBE_TEST=1  # Key: makes client connect to localhost!
export POD_NAMESPACE=discovery
export DYN_DISCOVERY_BACKEND=kubernetes
python3 -m your_client
```

This allows you to:
- ✅ Test with real Kubernetes resources
- ✅ Debug your metadata server locally
- ✅ See full discovery flow with actual metadata exchange
- ✅ Iterate quickly without deploying to K8s

The client discovers the pod from K8s but connects to `localhost:9000` for metadata!

## Next Steps

After these tests pass:
1. Test with real metadata servers using local testing mode (see above)
2. Test error handling (network failures, timeouts, etc.)
3. Test scale (100s of pods)
4. Test label selector edge cases
5. Add RBAC roles and test with restricted permissions

