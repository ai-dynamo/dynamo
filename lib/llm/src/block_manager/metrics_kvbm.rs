use dynamo_runtime::metrics::MetricsRegistry;
use prometheus::IntCounter;


#[derive(Clone, Debug)]
pub struct KvbmMetrics {
    pub offload_requests: IntCounter,
    pub save_kv_layer_requests: IntCounter,
}

impl KvbmMetrics {
    pub fn new(mr: &dyn MetricsRegistry) -> Self {
        let offload_requests = mr.create_intcounter("offload_requests", "The number of offload requests", &[]).unwrap();
        let save_kv_layer_requests = mr.create_intcounter("save_kv_layer_requests", "The number of save kv layer requests", &[]).unwrap();
        Self { offload_requests, save_kv_layer_requests }
    }
}
