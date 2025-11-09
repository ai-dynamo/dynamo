# Local Testing Mode

This guide explains how to test the Kubernetes discovery client with a **local metadata server** while watching **real Kubernetes resources**.

## Overview

The local testing mode allows you to:
- ✅ Watch real Kubernetes EndpointSlices
- ✅ Connect to a local metadata server (on localhost) instead of pod IPs
- ✅ Test the full discovery flow with your actual metadata implementation
- ✅ Debug and iterate quickly without deploying to Kubernetes

## How It Works

When `DYN_LOCAL_KUBE_TEST=1` is set:
1. The discovery client watches Kubernetes for EndpointSlices (as normal)
2. When a pod is discovered, it parses the pod name for a port number
3. If the pod name ends with `-<port>` (e.g., `dynamo-test-worker-8080`), it connects to `localhost:<port>` instead of the pod IP
4. Your local metadata server running on that port receives the request

## Setup

### 1. Create a Test Pod and Service

Create a pod and service with a specific port number:

```bash
cd k8s-test

# Create pod and service with default labels
./create-local-test-pod.sh 8080

# Or with custom Kubernetes namespace
./create-local-test-pod.sh 8080 my-k8s-namespace

# Or with custom Dynamo namespace and component labels
./create-local-test-pod.sh 8080 discovery hello_world backend
```

**Arguments:**
1. `port` - Port number (required) - used in pod name and for localhost connection
2. `k8s-namespace` - Kubernetes namespace (default: `discovery`)
3. `dynamo-namespace` - Value for `dynamo.nvidia.com/namespace` label (default: `test-namespace`)
4. `dynamo-component` - Value for `dynamo.nvidia.com/component` label (default: `test-component`)

This creates:
- A pod named `dynamo-test-worker-<port>`
- A service named `dynamo-test-service-<port>`
- An EndpointSlice (automatically created by Kubernetes for the service)

### 2. Start Your Local Metadata Server

Start your metadata server on the port you specified:

```bash
# Example with the system status server
cargo run --bin your-app -- --port 8080
```

Make sure your server exposes the `/metadata` endpoint that returns a JSON-serialized `DiscoveryMetadata` structure.

### 3. Run Tests in Local Mode

Set the environment variable and run your tests:

```bash
export DYN_LOCAL_KUBE_TEST=1
cargo test --test kube_client_integration test_watch_all_endpoints -- --ignored --nocapture
```

You should see logs like:
```
Local test mode: using localhost:8080 for pod dynamo-test-worker-8080
Fetching metadata from http://localhost:8080/metadata
```

## Multiple Local Servers

You can create multiple test pods with different ports and labels:

```bash
# Create pods for different components
./create-local-test-pod.sh 8080 discovery hello_world frontend
./create-local-test-pod.sh 8081 discovery hello_world backend
./create-local-test-pod.sh 8082 discovery hello_world worker
```

Then run multiple metadata servers on different ports:

```bash
# Terminal 1 - Frontend server
export PORT=8080
export POD_NAME=dynamo-test-worker-8080
export POD_NAMESPACE=discovery
your-server --component frontend

# Terminal 2 - Backend server
export PORT=8081
export POD_NAME=dynamo-test-worker-8081
export POD_NAMESPACE=discovery
your-server --component backend

# Terminal 3 - Worker server
export PORT=8082
export POD_NAME=dynamo-test-worker-8082
export POD_NAMESPACE=discovery
your-server --component worker
```

The discovery client will discover all three and connect to the appropriate localhost port for each!

## Pod Name Format

The pod name MUST end with `-<port>` where `<port>` is a valid port number:

✅ Valid:
- `dynamo-test-worker-8080`
- `my-service-9000`
- `test-pod-3000`

❌ Invalid:
- `dynamo-test-worker` (no port)
- `dynamo-test-worker-abc` (not a number)
- `8080-worker` (port not at the end)

The helper script automatically creates pods with the correct naming format.

## Example: Testing the Discovery Flow with hello_world

Here's a complete example using the `hello_world` app from the terminal output:

```bash
# 1. Create test pod with hello_world labels
cd k8s-test
./create-local-test-pod.sh 9000 discovery hello_world backend

# 2. Start your server (in another terminal)
cd ../examples/custom_backend/hello_world

# Set environment variables for the server
export PORT=9000
export DYN_SYSTEM_PORT=$PORT
export DYN_LOCAL_KUBE_TEST=1  # Not needed for server, but harmless
export POD_NAME=dynamo-test-worker-$PORT
export POD_NAMESPACE=discovery
export DYN_DISCOVERY_BACKEND=kubernetes

# Run the server
python3 -m hello_world

# 3. In another terminal, run the client
export PORT=9009  # Different port for client
export DYN_SYSTEM_PORT=$PORT
export DYN_LOCAL_KUBE_TEST=1  # IMPORTANT: Client needs this!
export POD_NAME=dynamo-test-worker-$PORT
export POD_NAMESPACE=discovery
export DYN_DISCOVERY_BACKEND=kubernetes

python3 -m client
```

You should see:
1. The server registers endpoint `hello_world/backend/generate` with its local metadata
2. The client discovers the pod `dynamo-test-worker-9000` from Kubernetes
3. The client connects to `http://localhost:9000/metadata` (not the pod IP!)
4. The server responds with its registered metadata
5. The client emits an `Added` event and can now make requests

This lets you:
- ✅ Debug both server and client locally
- ✅ See actual Kubernetes discovery in action
- ✅ Test with real metadata exchange
- ✅ Iterate quickly without container builds

## Cleanup

Delete test resources when done:

```bash
# Delete specific pod and service
kubectl delete pod/dynamo-test-worker-9000 --namespace=discovery
kubectl delete service/dynamo-test-service-9000 --namespace=discovery

# Or delete multiple at once
kubectl delete pod/dynamo-test-worker-8080 service/dynamo-test-service-8080 --namespace=discovery
kubectl delete pod/dynamo-test-worker-8081 service/dynamo-test-service-8081 --namespace=discovery
```

Or delete all local test resources at once:

```bash
kubectl delete pods,services -l app=dynamo-local-test --namespace=discovery
```

## Troubleshooting

### "Connection refused" to localhost

**Problem:** The discovery client can't connect to your local metadata server.

**Solution:**
- Ensure your metadata server is running on the correct port
- Check that the port matches the pod name (e.g., pod `...-8080` → server on port 8080)
- Verify your server exposes `/metadata` endpoint

### Pod name doesn't have a port

**Problem:** You created a pod without using the helper script and the name doesn't end with a port number.

**Solution:**
- Delete the pod: `kubectl delete pod/<pod-name>`
- Use the helper script: `./create-local-test-pod.sh 8080`
- Or manually create a pod with a name ending in `-<port>`

### Still connecting to pod IP instead of localhost

**Problem:** The environment variable isn't set.

**Solution:**
```bash
export DYN_LOCAL_KUBE_TEST=1
# Verify it's set
echo $DYN_LOCAL_KUBE_TEST  # Should print: 1
```

### Metadata format errors

**Problem:** Your local server returns data that doesn't match the expected `DiscoveryMetadata` format.

**Solution:**
Check the logs for JSON parsing errors. Your `/metadata` endpoint should return:
```json
{
  "endpoints": {
    "namespace/component/endpoint": {
      "Endpoint": {
        "namespace": "test-namespace",
        "component": "test-component",
        "endpoint": "test-endpoint",
        "instance_id": 12345,
        "transport": {"NatsTcp": "nats://localhost:4222"}
      }
    }
  },
  "model_cards": {}
}
```

## Production vs. Local Testing

| Mode | Environment Var | Connection | Use Case |
|------|----------------|------------|----------|
| Production | (none) | Pod IP:port | Real deployment |
| Mock | (test only) | No HTTP calls | Unit tests |
| Local Testing | `DYN_LOCAL_KUBE_TEST=1` | localhost:port | Integration testing with local server |

## Benefits

- 🚀 **Fast iteration**: No need to rebuild/redeploy containers
- 🐛 **Easy debugging**: Use debuggers, logging, etc. on your local server
- 🧪 **Full integration**: Test with real Kubernetes resources
- 💰 **Cost effective**: No cloud resources needed for testing
- ⚡ **Quick validation**: Test changes to metadata format instantly

