pub(crate) mod client;
pub(crate) mod protocol;
pub(crate) mod service;

use super::state::WorkerState;
use protocol::{FailedOnboardMessage, OffloadCompleteMessage, OnboardCompleteMessage};

const ONBOARD_COMPLETE_HANDLER: &str = "kvbm.connector.worker.onboard_complete";
const OFFLOAD_COMPLETE_HANDLER: &str = "kvbm.connector.worker.offload_complete";
const FAILED_ONBOARD_HANDLER: &str = "kvbm.connector.worker.failed_onboard";
const GET_LAYOUT_CONFIG_HANDLER: &str = "kvbm.connector.worker.get_layout_config";
const INITIALIZE_HANDLER: &str = "kvbm.connector.worker.initialize";
