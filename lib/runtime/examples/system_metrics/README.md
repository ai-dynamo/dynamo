# Generic Profiling for Request Handlers

This example demonstrates how to add automatic Prometheus metrics profiling to any request handler without modifying the handler code itself.

## Overview

The `RequestHandlerMetrics` system provides automatic profiling capabilities that are applied to all request handlers automatically. It automatically tracks:

- **Request Count**: Total number of requests processed
- **Request Duration**: Time spent processing each request
- **Request/Response Bytes**: Total bytes received and sent
- **Error Count**: Total number of errors encountered

Additionally, the example demonstrates how to add custom metrics with data bytes tracking in `MySystemStatsMetrics`.

## How It Works

**Automatic Metrics**: All request handlers automatically get profiling metrics without any code changes.

**Custom Metrics**: If you want to add custom metrics IN ADDITION to the automatic ones, you can use the `add_metrics` method:

```rust
use dynamo_runtime::pipeline::network::Ingress;

// Automatic profiling - no code changes needed!
let ingress = Ingress::for_engine(my_handler)?;

// Optional: Add custom metrics IN ADDITION to automatic ones
ingress.add_metrics(&endpoint)?;
```

The endpoint automatically provides proper labeling (dynamo_namespace, dynamo_component, dynamo_endpoint) for all metrics. These labels are prefixed with "dynamo_" to avoid collisions with Kubernetes and other monitoring system labels.

## Available Methods

The `Ingress` struct provides methods for metrics:

- **Automatic**: All request handlers get profiling metrics automatically
- `Ingress::add_metrics(&endpoint)` - Add custom metrics IN ADDITION to automatic ones (optional)

## Metrics Generated

### Automatic Metrics (No Code Changes Required)
The following Prometheus metrics are automatically created for all request handlers:

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
- `dynamo_component_concurrent_requests` - Number of requests currently being processed

### Custom Metrics (Optional)
- `dynamo_component_my_custom_bytes_processed_total` - Total data bytes processed by system handler (example)

### Labels
All metrics automatically include these labels from the endpoint:
- `dynamo_namespace` - The namespace name
- `dynamo_component` - The component name
- `dynamo_endpoint` - The endpoint name

These labels are prefixed with "dynamo_" to avoid collisions with Kubernetes and other monitoring system labels.

## Example Metrics Output

When the system is running, you'll see metrics from the /metrics HTTP path like this:

```prometheus
# HELP dynamo_component_concurrent_requests Number of requests currently being processed by request handler
# TYPE dynamo_component_concurrent_requests gauge
dynamo_component_concurrent_requests{dynamo_component="example_component",dynamo_endpoint="example_endpoint9881",dynamo_namespace="example_namespace"} 0

# HELP dynamo_component_my_custom_bytes_processed_total Example of a custom metric. Total number of data bytes processed by system handler
# TYPE dynamo_component_my_custom_bytes_processed_total counter
dynamo_component_my_custom_bytes_processed_total{dynamo_component="example_component",dynamo_endpoint="example_endpoint9881",dynamo_namespace="example_namespace"} 42

# HELP dynamo_component_request_bytes_total Total number of bytes received in requests by request handler
# TYPE dynamo_component_request_bytes_total counter
dynamo_component_request_bytes_total{dynamo_component="example_component",dynamo_endpoint="example_endpoint9881",dynamo_namespace="example_namespace"} 1098

# HELP dynamo_component_request_duration_seconds Time spent processing requests by request handler
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

# HELP dynamo_component_requests_total Total number of requests processed by request handler
# TYPE dynamo_component_requests_total counter
dynamo_component_requests_total{dynamo_component="example_component",dynamo_endpoint="example_endpoint9881",dynamo_namespace="example_namespace"} 3

# HELP dynamo_component_response_bytes_total Total number of bytes sent in responses by request handler
# TYPE dynamo_component_response_bytes_total counter
dynamo_component_response_bytes_total{dynamo_component="example_component",dynamo_endpoint="example_endpoint9881",dynamo_namespace="example_namespace"} 1917

# HELP uptime_seconds Total uptime of the DistributedRuntime in seconds
# TYPE uptime_seconds gauge
uptime_seconds{dynamo_namespace="http_server"} 1.8226759879999999
```

## Examples

### Example 1: Simple Handler with Automatic Profiling

```rust
struct SimpleHandler;

#[async_trait]
impl AsyncEngine<SingleIn<String>, ManyOut<Annotated<String>>, Error> for SimpleHandler {
    async fn generate(&self, input: SingleIn<String>) -> Result<ManyOut<Annotated<String>>> {
        // Your business logic here
        // No need to add any metrics code!
        Ok(ResponseStream::new(Box::pin(stream), ctx.context()))
    }
}

// Automatic profiling - no additional code needed!
let ingress = Ingress::for_engine(SimpleHandler::new())?;
```

### Example 2: Custom Handler with Data Bytes Tracking

```rust
struct RequestHandler {
    metrics: Option<Arc<MySystemStatsMetrics>>,
}

#[async_trait]
impl AsyncEngine<SingleIn<String>, ManyOut<Annotated<String>>, Error> for RequestHandler {
    async fn generate(&self, input: SingleIn<String>) -> Result<ManyOut<Annotated<String>>> {
        let (data, ctx) = input.into_parts();

        // Track data bytes processed (custom metric)
        if let Some(metrics) = &self.metrics {
            metrics.data_bytes_processed.inc_by(data.len() as u64);
        }

        // Your business logic here...

        Ok(ResponseStream::new(Box::pin(stream), ctx.context()))
    }
}

// Create custom metrics and handler
let system_metrics = MySystemStatsMetrics::from_endpoint(&endpoint)?;
let handler = RequestHandler::with_metrics(system_metrics);
let ingress = Ingress::for_engine(handler)?;

// Add custom metrics IN ADDITION to automatic ones
// You'll get both: automatic metrics (dynamo_component_requests_total, dynamo_component_request_duration_seconds, etc.)
// AND custom metrics (dynamo_component_my_custom_bytes_processed_total)
ingress.add_metrics(&endpoint)?;
```

## Benefits

1. **Zero Code Changes**: Existing handlers automatically get profiling metrics
2. **Simple API**: Just create an Ingress and you get metrics automatically
3. **Optional Custom Metrics**: Add custom metrics when needed
4. **Automatic Profiling**: Request count, duration, and error tracking out of the box
5. **Automatic Labeling**: Endpoint provides proper namespace/component/endpoint labels
6. **Performance**: Minimal overhead, metrics are only recorded when provided

## Running the Example

**Important**: You must set the `DYN_SYSTEM_PORT` environment variable to specify which port the HTTP server will run on.

```bash
# Run the system metrics example
DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=8081 cargo run --bin system_server
```
The server will start an HTTP server on the specified port (8081 in this example) that exposes the Prometheus metrics endpoint at `/metrics`.


To Run an actual LLM frontend + server (aggregated example), launch both of them. By default, the frontend listens to port 8080.
```
python -m dynamo.frontend &

DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=8081 python -m dynamo.vllm --model Qwen/Qwen3-0.6B --enforce-eager --no-enable-prefix-caching &
```
Then make curl requests to the frontend (see the [main README](../../../../README.md))

## Querying Metrics

Once running, you can query the metrics:

```bash
# Get all request handler metrics for components
curl http://localhost:8081/metrics | grep -E "dynamo_component"

# Get all frontend metrics
curl http://localhost:8080/metrics | grep -E "dynamo_frontend"
```