use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoutingMode {
    Off,       // default
    Auto,      // enforce only if model is "router" or missing
    Force,     // enforce even if a real model is set (SRE/testing)
    Shadow,    // never enforce; compute decision for telemetry
}

impl RoutingMode {
    pub fn from_header(val: Option<&str>) -> Self {
        match val.map(|v| v.to_ascii_lowercase()) {
            Some(ref s) if s == "auto" => RoutingMode::Auto,
            Some(ref s) if s == "force" => RoutingMode::Force,
            Some(ref s) if s == "shadow" => RoutingMode::Shadow,
            _ => RoutingMode::Off,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Target {
    OnPrem { model: String },
}

#[derive(Clone, Debug)]
pub struct RoutePlan {
    pub target: Target,
    pub rationale: &'static str,
    pub winner_label: String,
}

#[derive(Clone, Debug)]
pub struct RequestMeta<'a> {
    pub tenant: Option<&'a str>,
    pub region: Option<&'a str>,
    pub transport: &'a str,
    pub routing_mode: RoutingMode,
    pub model_field: Option<&'a str>,
    pub request_text: Option<&'a str>,
}

