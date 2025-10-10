# Generic Metrics for Component Endpoints

This example demonstrates the automatic metrics provided to component endpoints by default.

## Overview

Component endpoints are measured automatically when using the DistributedRuntime code. The DistributedRuntime uses the `MetricsRegistry` trait which provides automatic measurement capabilities that are applied to all component endpoints automatically. It automatically tracks:

- **Request Count**: Total number of requests processed
- **Request Duration**: Time spent processing each request
- **Request/Response Bytes**: Total bytes received and sent
- **Error Count**: Total number of errors encountered

Additionally, the example demonstrates how to add custom metrics with data bytes tracking.

## How It Works

**Automatic Metrics**: All component endpoints automatically get measurement metrics without any code changes.

**Custom Metrics**: If you want to add custom metrics IN ADDITION to the automatic ones, you can use the `add_metrics` method:

```rust
use dynamo_runtime::pipeline::network::Ingress;

// Automatic measurements - no code changes needed!
let ingress = Ingress::for_engine(my_handler)?;

// Optional: Add custom metrics IN ADDITION to automatic ones
ingress.add_metrics(&endpoint)?;
```

The endpoint automatically provides proper labeling (dynamo_namespace, dynamo_component, dynamo_endpoint) for all metrics. These labels are prefixed with "dynamo_" to avoid collisions with Kubernetes and other monitoring system labels.

## Available Methods

The `Ingress` struct provides methods for metrics:

- **Automatic**: All component endpoints get measurement metrics automatically
- `Ingress::add_metrics(&endpoint)` - Add custom metrics IN ADDITION to automatic ones (optional)

## Metrics Generated

### Automatic Metrics (No Code Changes Required)
The following Prometheus metrics are automatically created for all component endpoints:

### Counters
- `dynamo_component_requests_total` - Total requests processed
- `dynamo_component_request_bytes_total` - Total bytes received in requests
- `dynamo_component_response_bytes_total` - Total bytes sent in responses
- `dynamo_component_errors_total` - Total errors encountered (with error_type labels)

### Error Types
The `dynamo_component_errors_total` metric includes the following error types:
- `deserialization` - Errors parsing request messages
- `invalid_message` - Unexpected message format
- `response_stream` - Errors creating response streams
- `generate` - Errors in request processing
- `publish_response` - Errors publishing response data
- `publish_final` - Errors publishing final response

### Histograms
- `dynamo_component_request_duration_seconds` - Request processing time

### Gauges
- `dynamo_component_inflight_requests` - Number of requests currently being processed

### Custom Metrics (Optional)
- `dynamo_component_bytes_processed_total` - Total data bytes processed by system handler (example)

### Labels
All metrics automatically include these labels from the endpoint:
- `dynamo_namespace` - The namespace name
- `dynamo_component` - The component name
- `dynamo_endpoint` - The endpoint name

These labels are prefixed with "dynamo_" to avoid collisions with Kubernetes and other monitoring system labels.

## Example Metrics Output

When the system is running, you'll see metrics from http://<ip>:<port>/metrics like this:

```prometheus
# HELP dynamo_component_inflight_requests Number of requests currently being processed by component endpoint
# TYPE dynamo_component_inflight_requests gauge
dynamo_component_inflight_requests{dynamo_component="example_component",dynamo_endpoint="example_endpoint9881",dynamo_namespace="example_namespace"} 0

# HELP dynamo_component_bytes_processed_total Example of a custom metric. Total number of data bytes processed by system handler
# TYPE dynamo_component_bytes_processed_total counter
dynamo_component_bytes_processed_total{dynamo_component="example_component",dynamo_endpoint="example_endpoint9881",dynamo_namespace="example_namespace"} 42

# HELP dynamo_component_request_bytes_total Total number of bytes received in requests by component endpoint
# TYPE dynamo_component_request_bytes_total counter
dynamo_component_request_bytes_total{dynamo_component="example_component",dynamo_endpoint="example_endpoint9881",dynamo_namespace="example_namespace"} 1098

# HELP dynamo_component_request_duration_seconds Time spent processing requests by component endpoint
# TYPE dynamo_component_request_duration_seconds histogram
dynamo_component_request_duration_seconds_bucket{dynamo_component="example_component",dynamo_endpoint="example_endpoint9881",dynamo_namespace="example_namespace",le="0.005"} 3
dynamo_component_request_duration_seconds_bucket{dynamo_component="example_component",dynamo_endpoint="example_endpoint9881",dynamo_namespace="example_namespace",le="0.01"} 3
dynamo_component_request_duration_seconds_bucket{dynamo_component="example_component",dynamo_endpoint="example_endpoint9881",dynamo_namespace="example_namespace",le="0.025"} 3
dynamo_component_request_duration_seconds_bucket{dynamo_component="example_component",dynamo_endpoint="example_endpoint9881",dynamo_namespace="example_namespace",le="0.05"} 3
dynamo_component_request_duration_seconds_bucket{dynamo_component="example_component",dynamo_endpoint="example_endpoint9881",dynamo_namespace="example_namespace",le="0.1"} 3
dynamo_component_request_duration_seconds_bucket{dynamo_component="example_component",dynamo_endpoint="example_endpoint9881",dynamo_namespace="example_namespace",le="0.25"} 3
dynamo_component_request_duration_seconds_bucket{dynamo_component="example_component",dynamo_endpoint="example_endpoint9881",dynamo_namespace="example_namespace",le="0.5"} 3
dynamo_component_request_duration_seconds_bucket{dynamo_component="example_component",dynamo_endpoint="example_endpoint9881",dynamo_namespace="example_namespace",le="1"} 3
dynamo_component_request_duration_seconds_bucket{dynamo_component="example_component",dynamo_endpoint="example_endpoint9881",dynamo_namespace="example_namespace",le="2.5"} 3
dynamo_component_request_duration_seconds_bucket{dynamo_component="example_component",dynamo_endpoint="example_endpoint9881",dynamo_namespace="example_namespace",le="5"} 3
dynamo_component_request_duration_seconds_bucket{dynamo_component="example_component",dynamo_endpoint="example_endpoint9881",dynamo_namespace="example_namespace",le="10"} 3
dynamo_component_request_duration_seconds_bucket{dynamo_component="example_component",dynamo_endpoint="example_endpoint9881",dynamo_namespace="example_namespace",le="+Inf"} 3
dynamo_component_request_duration_seconds_sum{dynamo_component="example_component",dynamo_endpoint="example_endpoint9881",dynamo_namespace="example_namespace"} 0.00048793700000000003
dynamo_component_request_duration_seconds_count{dynamo_component="example_component",dynamo_endpoint="example_endpoint9881",dynamo_namespace="example_namespace"} 3

# HELP dynamo_component_requests_total Total number of requests processed by component endpoint
# TYPE dynamo_component_requests_total counter
dynamo_component_requests_total{dynamo_component="example_component",dynamo_endpoint="example_endpoint9881",dynamo_namespace="example_namespace"} 3

# HELP dynamo_component_response_bytes_total Total number of bytes sent in responses by component endpoint
# TYPE dynamo_component_response_bytes_total counter
dynamo_component_response_bytes_total{dynamo_component="example_component",dynamo_endpoint="example_endpoint9881",dynamo_namespace="example_namespace"} 1917

# HELP uptime_seconds Total uptime of the DistributedRuntime in seconds
# TYPE uptime_seconds gauge
uptime_seconds{dynamo_namespace="system_status_server"} 1.8226759879999999
```

## Example

### Component Endpoint with Automatic Measurements and Optional Custom Metrics

```rust
struct RequestHandler {
    metrics: Option<Arc<CustomMetrics>>,
}

#[async_trait]
impl AsyncEngine<SingleIn<String>, ManyOut<Annotated<String>>, Error> for RequestHandler {
    async fn generate(&self, input: SingleIn<String>) -> Result<ManyOut<Annotated<String>>> {
        let (data, ctx) = input.into_parts();

        // Optional: Track custom metrics
        if let Some(metrics) = &self.metrics {
            metrics.data_bytes_processed.inc_by(data.len() as u64);
        }

        // Your business logic here...
        // No need to add any automatic measurement code!

        Ok(ResponseStream::new(Box::pin(stream), ctx.context()))
    }
}

// Create handler (with or without custom metrics)
let handler = if enable_custom_metrics {
    let custom_metrics = CustomMetrics::from_endpoint(&endpoint)?;
    RequestHandler::with_metrics(custom_metrics)
} else {
    RequestHandler::new()
};

// Automatic measurements - no additional code needed!
let ingress = Ingress::for_engine(handler)?;

// Optional: Add custom metrics IN ADDITION to automatic ones
if enable_custom_metrics {
    ingress.add_metrics(&endpoint)?;
}

// Endpoint code to add ingress to the handler below...
```

## Benefits

1. **Little/No Code Changes**: Existing handlers automatically get measurement metrics, and easy to add custom metrics for your particular application.
2. **Simple API**: Simply swap out Prometheus constructors with one of the endpoint's factory methods.
3. **Automatic Measurements**: Request count, duration, and error tracking out of the box for component endpoints.
4. **Automatic Labeling**: Endpoint provides proper namespace/component/endpoint labels

## Running the Example

**Important**: You must set the `DYN_SYSTEM_PORT` environment variable to specify which port the system status server will listen on.

```bash
# Run the system metrics example
DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=8081 cargo run --bin system_server
```
The server will start an system status server on the specified port (8081 in this example) that exposes the Prometheus metrics endpoint at `/metrics`.


To Run an actual LLM frontend + server (aggregated example), launch both of them. By default, the frontend listens to port 8000.
```
python -m dynamo.frontend &

DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=8081 python -m dynamo.vllm --model Qwen/Qwen3-0.6B --enforce-eager --no-enable-prefix-caching &
```
Then make curl requests to the frontend (see the [main README](../../../../README.md))

## Querying Metrics

Once running, you can query the metrics:

```bash
# Get all component endpoint metrics for components
curl http://localhost:8081/metrics | grep -E "dynamo_component"

# Get all frontend metrics
curl http://localhost:8000/metrics | grep -E "dynamo_frontend"
```
