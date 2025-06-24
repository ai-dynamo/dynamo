//! Distributed Layout

use super::{nixl::*, *};
use crate::block_manager::storage::nixl::NixlStorage;

use serde::{Deserialize, Serialize};
use std::sync::Arc;

// /// Distributed Configuration
// #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
// pub struct DistributedConfig {
//     /// Number of workers
//     pub num_workers: usize,

//     /// Tensor Parallel Size
//     pub tp_size: usize,

//     /// Pipeline Parallel Size
//     pub pp_size: usize,

//     /// Layouts for each set of layers in the model
//     /// A fully symmetric model will have one entry
//     pub layouts: Vec<Arc<dyn BlockLayout<StorageType = NixlStorage>>>,
// }
