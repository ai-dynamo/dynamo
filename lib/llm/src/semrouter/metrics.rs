use once_cell::sync::Lazy;
use prometheus::{register_histogram_vec, register_int_counter_vec, HistogramVec, IntCounterVec};

pub static ROUTE_DECISIONS: Lazy<IntCounterVec> = Lazy::new(|| {
    register_int_counter_vec!(
        "semantic_route_decisions_total",
        "Route decisions made by semantic router",
        &["route", "target", "rationale", "winner", "transport"]
    )
    .expect("Failed to register semantic_route_decisions_total metric")
});

pub static CLASSIFIER_LATENCY: Lazy<HistogramVec> = Lazy::new(|| {
    register_histogram_vec!(
        "semantic_classifier_latency_ms",
        "Latency of classifier inference",
        &["transport"]
    )
    .expect("Failed to register semantic_classifier_latency_ms metric")
});

