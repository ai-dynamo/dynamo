# Generic Profiling for Work Handlers

This example demonstrates how to add automatic Prometheus metrics profiling to any work handler without modifying the handler code itself.

## Overview

The `WorkHandlerMetrics` system provides automatic profiling capabilities that are applied to all work handlers automatically. It automatically tracks:

- **Request Count**: Total number of requests processed
- **Request Duration**: Time spent processing each request
- **Request/Response Bytes**: Total bytes received and sent
- **Error Count**: Total number of errors encountered

Additionally, the example demonstrates how to add custom metrics with data bytes tracking in `MySystemStatsMetrics`.

## How It Works

**Automatic Metrics**: All work handlers automatically get profiling metrics without any code changes.

**Custom Metrics**: If you want to add custom metrics IN ADDITION to the automatic ones, you can use the `add_metrics` method:

```rust
use dynamo_runtime::pipeline::network::Ingress;

// Automatic profiling - no code changes needed!
let ingress = Ingress::for_engine(my_handler)?;

// Optional: Add custom metrics IN ADDITION to automatic ones
ingress.add_metrics(&endpoint)?;
```

The endpoint automatically provides proper labeling (namespace, component, endpoint) for all metrics.

## Available Methods

The `Ingress` struct provides methods for metrics:

- **Automatic**: All handlers get profiling metrics automatically
- `Ingress::add_metrics(&endpoint)` - Add custom metrics IN ADDITION to automatic ones (optional)

## Metrics Generated

### Automatic Metrics (No Code Changes Required)
The following Prometheus metrics are automatically created for all work handlers:

### Counters
- `requests_total` - Total requests processed
- `request_bytes_total` - Total bytes received in requests
- `response_bytes_total` - Total bytes sent in responses
- `errors_total` - Total errors encountered (with error_type labels)

### Error Types
The `errors_total` metric includes the following error types:
- `deserialization` - Errors parsing request messages
- `invalid_message` - Unexpected message format
- `response_stream` - Errors creating response streams
- `generate` - Errors in request processing
- `publish_response` - Errors publishing response data
- `publish_final` - Errors publishing final response

### Histograms
- `request_duration_seconds` - Request processing time

### Gauges
- `concurrent_requests` - Number of requests currently being processed

### Custom Metrics (Optional)
- `my_custom_bytes_processed_total` - Total data bytes processed by system handler (example)

### Labels
All metrics automatically include these labels from the endpoint:
- `namespace` - The namespace name
- `component` - The component name
- `endpoint` - The endpoint name

## Example Metrics Output

When the system is running, you'll see metrics from the /metrics HTTP path like this:

```prometheus
# HELP concurrent_requests Number of requests currently being processed by work handler
# TYPE concurrent_requests gauge
concurrent_requests{component="mycomponent",endpoint="myendpoint4598",namespace="mynamespace"} 0

# HELP my_custom_bytes_processed_total Example of a custom metric. Total number of data bytes processed by system handler
# TYPE my_custom_bytes_processed_total counter
my_custom_bytes_processed_total{component="mycomponent",endpoint="myendpoint4598",namespace="mynamespace"} 42

# HELP request_bytes_total Total number of bytes received in requests by work handler
# TYPE request_bytes_total counter
request_bytes_total{component="mycomponent",endpoint="myendpoint4598",namespace="mynamespace"} 1098

# HELP request_duration_seconds Time spent processing requests by work handler
# TYPE request_duration_seconds histogram
request_duration_seconds_bucket{component="mycomponent",endpoint="myendpoint4598",namespace="mynamespace",le="0.005"} 3
request_duration_seconds_bucket{component="mycomponent",endpoint="myendpoint4598",namespace="mynamespace",le="0.01"} 3
request_duration_seconds_bucket{component="mycomponent",endpoint="myendpoint4598",namespace="mynamespace",le="0.025"} 3
request_duration_seconds_bucket{component="mycomponent",endpoint="myendpoint4598",namespace="mynamespace",le="0.05"} 3
request_duration_seconds_bucket{component="mycomponent",endpoint="myendpoint4598",namespace="mynamespace",le="0.1"} 3
request_duration_seconds_bucket{component="mycomponent",endpoint="myendpoint4598",namespace="mynamespace",le="0.25"} 3
request_duration_seconds_bucket{component="mycomponent",endpoint="myendpoint4598",namespace="mynamespace",le="0.5"} 3
request_duration_seconds_bucket{component="mycomponent",endpoint="myendpoint4598",namespace="mynamespace",le="1"} 3
request_duration_seconds_bucket{component="mycomponent",endpoint="myendpoint4598",namespace="mynamespace",le="2.5"} 3
request_duration_seconds_bucket{component="mycomponent",endpoint="myendpoint4598",namespace="mynamespace",le="5"} 3
request_duration_seconds_bucket{component="mycomponent",endpoint="myendpoint4598",namespace="mynamespace",le="10"} 3
request_duration_seconds_bucket{component="mycomponent",endpoint="myendpoint4598",namespace="mynamespace",le="+Inf"} 3
request_duration_seconds_sum{component="mycomponent",endpoint="myendpoint4598",namespace="mynamespace"} 0.00048793700000000003
request_duration_seconds_count{component="mycomponent",endpoint="myendpoint4598",namespace="mynamespace"} 3

# HELP requests_total Total number of requests processed by work handler
# TYPE requests_total counter
requests_total{component="mycomponent",endpoint="myendpoint4598",namespace="mynamespace"} 3

# HELP response_bytes_total Total number of bytes sent in responses by work handler
# TYPE response_bytes_total counter
response_bytes_total{component="mycomponent",endpoint="myendpoint4598",namespace="mynamespace"} 1917

# HELP uptime_seconds Total uptime of the DistributedRuntime in seconds
# TYPE uptime_seconds gauge
uptime_seconds{namespace="http_server"} 1.8226759879999999
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
// You'll get both: automatic metrics (requests_total, request_duration_seconds, etc.)
// AND custom metrics (my_custom_bytes_processed_total)
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
export DYN_LOG=1 DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=8081
cargo run --bin system_server
```

The server will start an HTTP server on the specified port (9091 in this example) that exposes the Prometheus metrics endpoint at `/metrics`.

## Querying Metrics

Once running, you can query the metrics:

```bash
# Get all ingress metrics
curl http://localhost:9091/metrics | grep ingress_

# Get request duration histogram
curl http://localhost:9091/metrics | grep 'ingress_request_duration_seconds'

### View Metrics
```bash
curl http://localhost:8081/metrics
```

Example output:
```
# HELP service_request_duration_seconds Time spent processing requests
# TYPE service_request_duration_seconds histogram
service_request_duration_seconds_bucket{component="component",endpoint="endpoint",namespace="system",service="backend",le="0.005"} 2
service_request_duration_seconds_bucket{component="component",endpoint="endpoint",namespace="system",service="backend",le="0.01"} 2
service_request_duration_seconds_bucket{component="component",endpoint="endpoint",namespace="system",service="backend",le="0.025"} 2
service_request_duration_seconds_bucket{component="component",endpoint="endpoint",namespace="system",service="backend",le="0.05"} 2
service_request_duration_seconds_bucket{component="component",endpoint="endpoint",namespace="system",service="backend",le="0.1"} 2
service_request_duration_seconds_bucket{component="component",endpoint="endpoint",namespace="system",service="backend",le="0.25"} 2
service_request_duration_seconds_bucket{component="component",endpoint="endpoint",namespace="system",service="backend",le="0.5"} 2
service_request_duration_seconds_bucket{component="component",endpoint="endpoint",namespace="system",service="backend",le="1"} 2
service_request_duration_seconds_bucket{component="component",endpoint="endpoint",namespace="system",service="backend",le="2.5"} 2
service_request_duration_seconds_bucket{component="component",endpoint="endpoint",namespace="system",service="backend",le="5"} 2
service_request_duration_seconds_bucket{component="component",endpoint="endpoint",namespace="system",service="backend",le="10"} 2
service_request_duration_seconds_bucket{component="component",endpoint="endpoint",namespace="system",service="backend",le="+Inf"} 2
service_request_duration_seconds_sum{component="component",endpoint="endpoint",namespace="system",service="backend"} 0.000022239000000000002
service_request_duration_seconds_count{component="component",endpoint="endpoint",namespace="system",service="backend"} 2
# HELP service_requests_total Total number of requests processed
# TYPE service_requests_total counter
service_requests_total{component="component",endpoint="endpoint",namespace="system",service="backend"} 2
# HELP uptime_seconds Total uptime of the DistributedRuntime in seconds
# TYPE uptime_seconds gauge
uptime_seconds{namespace="http_server"} 725.997013676
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DYN_LOG` | Enable logging | `0` |
| `DYN_SYSTEM_ENABLED` | Enable system metrics | `false` |
| `DYN_SYSTEM_PORT` | HTTP server port | `8081` |

## Metrics

- `service_requests_total`: Request counter
- `service_request_duration_seconds`: Request duration histogram
- `uptime_seconds`: Server uptime gauge

This provides automatic context and grouping for all metrics without manual configuration.

## Troubleshooting

- **Port in use**: Change `DYN_SYSTEM_PORT`
- **Connection refused**: Ensure server is running first
- **No metrics**: Verify `DYN_SYSTEM_ENABLED=true`
