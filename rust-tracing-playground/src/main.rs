use opentelemetry::global;
use opentelemetry::trace::{TraceContextExt};
use opentelemetry::propagation::{Extractor, Injector, TextMapPropagator};
use opentelemetry_otlp::WithExportConfig;
use std::collections::HashMap;
use std::time::Duration;
use tracing::{info, instrument, Span, Event, Subscriber};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, Layer, registry::LookupSpan};
use tracing_opentelemetry::OpenTelemetrySpanExt;
use std::fmt;

/// A custom layer that only prints event messages
struct CustomLayer;

impl<S> Layer<S> for CustomLayer 
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_new_span(&self, attrs: &tracing::span::Attributes<'_>, id: &tracing::span::Id, ctx: tracing_subscriber::layer::Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            // Method 2: Get from current span context (alternative approach)
            let current_span = Span::current();
            let otel_context = current_span.context();
            let span_ref = otel_context.span();
            let span_context = span_ref.span_context();
            let current_trace_id = format!("{}", span_context.trace_id());
            let current_span_id = format!("{}", span_context.span_id());

            let span_name = span.name();
            // Get the OpenTelemetry data from the span extensions
            let extensions = span.extensions();
            let (trace_id, span_id) = if let Some(otel_data) = extensions.get::<tracing_opentelemetry::OtelData>() {
                let trace_id = otel_data.builder.trace_id
                    .map(|id| format!("{:?}", id))
                    .unwrap_or_else(|| "no_trace".to_string());
                let span_id = otel_data.builder.span_id
                    .map(|id| format!("{:?}", id))
                    .unwrap_or_else(|| "no_span".to_string());
                (trace_id, span_id)
            } else {
                ("no_otel_data".to_string(), "no_otel_data".to_string())
            };
            
            // Also check the span attributes
            let mut visitor = MessageVisitor::default();
            attrs.record(&mut visitor);
            let attrs_info = visitor.message.unwrap_or_else(|| "no_attrs".to_string());
            
            println!("[NEW_SPAN] span_name={}, attrs={}, builder_trace_id={}, builder_span_id={}, current_trace_id={}, current_span_id={}", 
                span_name, attrs_info, trace_id, span_id, current_trace_id, current_span_id);
        }
    }

    fn on_enter(&self, id: &tracing::span::Id, ctx: tracing_subscriber::layer::Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            // Method 2: Get from current span context (alternative approach)
            let current_span = Span::current();
            let otel_context = current_span.context();
            let span_ref = otel_context.span();
            let span_context = span_ref.span_context();
            let current_trace_id = format!("{}", span_context.trace_id());
            let current_span_id = format!("{}", span_context.span_id());


            let span_name = span.name();
            // Get the OpenTelemetry data from the span extensions
            let extensions = span.extensions();
            let (trace_id, span_id) = if let Some(otel_data) = extensions.get::<tracing_opentelemetry::OtelData>() {
                let trace_id = otel_data.builder.trace_id
                    .map(|id| format!("{:?}", id))
                    .unwrap_or_else(|| "no_trace".to_string());
                let span_id = otel_data.builder.span_id
                    .map(|id| format!("{:?}", id))
                    .unwrap_or_else(|| "no_span".to_string());
                (trace_id, span_id)
            } else {
                ("no_otel_data".to_string(), "no_otel_data".to_string())
            };
            println!("[ENTER] span_name={}, builder_trace_id={}, builder_span_id={}, current_trace_id={}, current_span_id={}", 
                span_name, trace_id, span_id, current_trace_id, current_span_id);
            
            // println!("ENTER [trace_id={}, span_id={}] span: {}", trace_id, span_id, span_name);
        }
    }
    
    fn on_event(&self, event: &Event<'_>, ctx: tracing_subscriber::layer::Context<'_, S>) {
        let current_span = Span::current();
        let otel_context = current_span.context();
        let span_ref = otel_context.span();
        let span_context = span_ref.span_context();
        let current_trace_id = format!("{}", span_context.trace_id());
        let current_span_id = format!("{}", span_context.span_id());

        // Get the message field from the event
        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);
        
        // Get the span context if it exists
        let (trace_id, span_id) = if let Some(span) = ctx.lookup_current() {
            let extensions = span.extensions();
            if let Some(otel_data) = extensions.get::<tracing_opentelemetry::OtelData>() {
                let trace_id = otel_data.builder.trace_id
                    .map(|id| format!("{:?}", id))
                    .unwrap_or_else(|| "no_trace".to_string());
                let span_id = otel_data.builder.span_id
                    .map(|id| format!("{:?}", id))
                    .unwrap_or_else(|| "no_span".to_string());
                (trace_id, span_id)
            } else {
                ("no_otel_data".to_string(), "no_otel_data".to_string())
            }
        } else {
            ("root".to_string(), "root".to_string())
        };

        // Print the message with trace context if we found one
        if let Some(message) = visitor.message {
            println!("EVENT [trace_id={}, span_id={} current_trace_id {}, current_span_id {}] {}", trace_id, span_id, current_trace_id, current_span_id, message);
        }
    }
}

#[derive(Default)]
struct MessageVisitor {
    message: Option<String>,
}

impl tracing::field::Visit for MessageVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn fmt::Debug) {
        if field.name() == "message" {
            self.message = Some(format!("{:?}", value));
        }
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "message" {
            self.message = Some(value.to_string());
        }
    }
}

#[instrument]
async fn simulate_work(task_name: &str, duration_ms: u64) {
    info!(task = task_name, "Starting task");
    tokio::time::sleep(Duration::from_millis(duration_ms)).await;
    
    // Simulate some nested work
    for i in 0..1 {
        simulate_nested_task(i, duration_ms / 3).await;
    }
    
    info!(task = task_name, "Task completed");
}

#[instrument]
async fn simulate_nested_task(id: u32, duration_ms: u64) {
    let current_span = tracing::Span::current();
    let otel_context = current_span.context();
    let span = otel_context.span();
    let span_context = span.span_context();
    
    info!(
        task_id = id,
        trace_id = %span_context.trace_id(),
        span_id = %span_context.span_id(),
        "Processing subtask"
    );
    
    tokio::time::sleep(Duration::from_millis(duration_ms)).await;
    
    if id == 1 {
        // Simulate an error condition occasionally
        tracing::warn!(
            task_id = id, 
            trace_id = %span_context.trace_id(),
            span_id = %span_context.span_id(),
            "Simulated warning in subtask");
    }
}

#[instrument]
async fn process_request(request_id: u32) {
    info!(request_id = request_id, "Processing new request");
    
    // Simulate different types of work
    simulate_work("database_query", 200).await;
    // simulate_work("external_api_call", 300).await;
    // simulate_work("data_processing", 150).await;
    
    info!(request_id = request_id, "Request processing completed");
}

/// Simulate extracting trace context from HTTP headers (incoming request)
fn extract_trace_context_from_headers(headers: &HashMap<String, String>) -> opentelemetry::Context {
    // Create an extractor that reads from our headers map
    struct HeaderExtractor<'a>(&'a HashMap<String, String>);
    
    impl<'a> Extractor for HeaderExtractor<'a> {
        fn get(&self, key: &str) -> Option<&str> {
            self.0.get(key).map(|v| v.as_str())
        }

        fn keys(&self) -> Vec<&str> {
            self.0.keys().map(|k| k.as_str()).collect()
        }
    }

    let extractor = HeaderExtractor(headers);
    let propagator = opentelemetry::sdk::propagation::TraceContextPropagator::new();
    
    // Extract the context from headers
    propagator.extract(&extractor)
}

/// Simulate injecting trace context into HTTP headers (outgoing request)
fn inject_trace_context_into_headers(context: &opentelemetry::Context) -> HashMap<String, String> {
    let mut headers = HashMap::new();
    
    // Create an injector that writes to our headers map
    struct HeaderInjector<'a>(&'a mut HashMap<String, String>);
    
    impl<'a> Injector for HeaderInjector<'a> {
        fn set(&mut self, key: &str, value: String) {
            self.0.insert(key.to_string(), value);
        }
    }

    let mut injector = HeaderInjector(&mut headers);
    let propagator = opentelemetry::sdk::propagation::TraceContextPropagator::new();
    
    // Inject the current context into headers
    propagator.inject_context(context, &mut injector);
    
    headers
}

/// Process a request as if it came from another service (with trace context in headers)
// #[instrument]
async fn process_request_from_service(request_id: u32, headers: HashMap<String, String>) {
    // Extract trace context from incoming headers
    let parent_context = extract_trace_context_from_headers(&headers);
    // parent_context.span().span_context().parent_id();
    
    // Create a span with the extracted context as parent
    let span = tracing::info_span!("process_request_from_service", request_id = request_id);

    // the span itself can be set with 
    span.set_parent(parent_context);
    
    let _enter = span.enter();
    
    info!(request_id = request_id, "Processing request from external service");
    
    // Log the trace context to verify it was propagated correctly
    let current_span = Span::current();
    let otel_context = current_span.context();
    let span_ref = otel_context.span();
    let span_context = span_ref.span_context();
    
    info!(
        request_id = request_id,
        trace_id = %span_context.trace_id(),
        span_id = %span_context.span_id(),
        "Received trace context from headers"
    );
    
    // Continue with normal processing - these will be part of the same trace
    simulate_work("database_query", 200).await;
    simulate_work("external_api_call", 300).await;
    simulate_work("data_processing", 150).await;
    
    info!(request_id = request_id, "Request from external service completed");
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create OTLP exporter that sends to Tempo
    let otlp_exporter = opentelemetry_otlp::new_exporter()
        .tonic()
        .with_endpoint("http://localhost:4317");

    let tracer = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(otlp_exporter)
        .with_trace_config(
            opentelemetry_sdk::trace::config()
                .with_resource(opentelemetry_sdk::Resource::new(vec![
                    opentelemetry::KeyValue::new("service.name", "rust-tracing-playground"),
                    opentelemetry::KeyValue::new("service.version", "1.0.0"),
                ]))
        )
        .install_batch(opentelemetry_sdk::runtime::Tokio)?;

    // Initialize tracing subscriber
    tracing_subscriber::registry()
        .with(tracing_opentelemetry::layer().with_tracer(tracer))
        .with(tracing_subscriber::filter::LevelFilter::INFO)
        .with(CustomLayer)
        .init();

    info!("🚀 Starting Rust Tracing Playground with OTLP export to Tempo");

    // Generate 2 separate requests with distinct trace IDs
    // Each process_request call will create its own root trace
    let mut propagated_headers = HashMap::new();
    
    {
        let request_span = tracing::info_span!("request_1");

        let otel_context = request_span.context();
        let span_ref = otel_context.span();
        let span_context = span_ref.span_context();
        println!("got here");
        println!("Checking if this is accessible before entry: span_context: trace_id={}, span_id={}", span_context.trace_id(), span_context.span_id());


        let _enter = request_span.enter();
        info!("Processing first request");
        process_request(1).await;
        
        // Simulate making an outbound call - inject trace context into headers
        let current_span = Span::current();
        let otel_context: opentelemetry::Context = current_span.context();
        propagated_headers = inject_trace_context_into_headers(&otel_context);
        
        info!("Injected trace context into headers for service boundary crossing");
        for (key, value) in &propagated_headers {
            info!(header_key = %key, header_value = %value, "Propagation header");
        }

        // Simulate receiving a request from another service with trace context
        info!("Simulating request from external service with propagated trace context");
    }
    // can you print the hashmap using println
    for (key, value) in &propagated_headers {
        println!("Propagation header: key={}, value={}", key, value);
    }
    process_request_from_service(3, propagated_headers).await;


    // Wait between requests to clearly separate them
    tokio::time::sleep(Duration::from_secs(1)).await;
    
    // {
    //     let request_span = tracing::info_span!("request_2");
    //     let _enter = request_span.enter();
    //     info!("Processing second request");
    //     process_request(2).await;
    // }

    // Wait a bit more
    tokio::time::sleep(Duration::from_secs(1)).await;

    info!("🏁 Both requests completed, shutting down");
    
    // Ensure all traces are exported before shutdown
    global::shutdown_tracer_provider();
    
    // Give some time for final export
    tokio::time::sleep(Duration::from_secs(2)).await;
    
    Ok(())
}