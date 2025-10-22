pub mod classifier;
pub mod config;
pub mod hook;
pub mod metrics;
pub mod policy;
pub mod request_processor;
pub mod routing;
pub mod types;

pub use classifier::Classifier;
pub use classifier::mock::MockClassifier;
#[cfg(feature = "onnx-classifier")]
pub use classifier::onnx::OnnxClassifier;
#[cfg(feature = "candle-classifier")]
pub use classifier::candle::CandleClassifier;
pub use config::PolicyConfig;
pub use hook::{RouteDecision, SemRouter};
pub use policy::CategoryPolicy;
pub use request_processor::process_chat_request;
pub use routing::{apply_routing, apply_routing_direct, extract_chat_text};
pub use types::{RequestMeta, RoutePlan, RoutingMode, Target};

