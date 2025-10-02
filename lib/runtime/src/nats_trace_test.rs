// run with `cargo run --bin nats_trace_test` after exporting the env variable `DYN_LOGGING_JSONL=1`
// should be in ./bin
use dynamo_runtime::logging::{init, inject_current_trace_into_nats_headers, create_span_from_nats_headers};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set environment variable for JSON logging
    // std::env::set_var("DYN_LOGGING_JSONL", "1");
    
    println!("🚀 Starting NATS header trace propagation test...");

    init();
    
    // Run the test scenario
    for i in 1..=3 {
        println!("🔄 Running scenario {}...", i);
        nats_trace_scenario().await;
        
        // Small delay between scenarios
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    
    println!("⏳ Waiting for OTLP traces to flush...");
    
    // Give time for OTLP batches to flush
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    
    // Flush OTLP traces before ending
    opentelemetry::global::shutdown_tracer_provider();
    
    // Additional wait to ensure everything is sent
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    
    println!("✅ Test completed! Check Tempo for traces.");
    
    Ok(())
}

/// Test scenario that simulates distributed tracing across NATS message boundaries
async fn nats_trace_scenario() {
    let starting_span = tracing::info_span!("starting_span");
    let _starting_enter = starting_span.enter();

    // 1. Create a span with a nested span and events
    tracing::info!(message = "Starting original request processing");
    
    // Create nested span within original request
    let nested_span = tracing::info_span!("nested_processing");
    let _nested_enter = nested_span.enter();
    tracing::info!(message = "Processing in nested span");
    tracing::debug!(processing_step = "validation", status = "complete");
    drop(_nested_enter);
    drop(nested_span);
    
    // 2. Populate NATS headers with the current trace context
    let mut nats_headers = async_nats::HeaderMap::new();
    inject_current_trace_into_nats_headers(&mut nats_headers);
    
    tracing::info!(message = "Injected trace context into NATS headers");
    
    // Log the headers for debugging (in a real scenario, these would be sent over NATS)
    if let Some(traceparent) = nats_headers.get("traceparent") {
        tracing::debug!(traceparent = traceparent.as_str(), message = "NATS header populated");
        println!("📋 NATS traceparent header: {}", traceparent.as_str());
    }
    
    // Drop the starting span to simulate process boundary
    // drop(_starting_enter);
    // drop(starting_span);
    
    println!("🔄 Simulating process boundary...");
    
    // 3. Simulate process boundary - create new span from NATS headers
    simulate_process_boundary(&nats_headers).await;
}

/// Simulates receiving a NATS message in a different process/service
async fn simulate_process_boundary(headers: &async_nats::HeaderMap) {
    // Extract context and create new span as entry point for this "service"
    let boundary_span = create_span_from_nats_headers(headers);
    let _boundary_enter = boundary_span.enter();
    
    tracing::info!(message = "Entered new process boundary from NATS headers");
    
    // 4. Create events and nested span within the new process boundary
    tracing::info!(message = "Processing message in new service");
    tracing::debug!(service = "message_processor", action = "received");
    
    // Create nested processing within the boundary span
    let processing_span = tracing::info_span!("message_processing");
    let _processing_enter = processing_span.enter();
    
    tracing::info!(message = "Deep processing in boundary service");
    tracing::debug!(processing_phase = "transformation", result = "success");
    
    // Simulate some async work
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    
    tracing::info!(message = "Completed processing in boundary service");
    
    drop(_processing_enter);
    drop(processing_span);
    drop(_boundary_enter);
    drop(boundary_span);
}
