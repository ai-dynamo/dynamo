use std::collections::HashMap;

use crate::block_manager::block::nixl::{NixlBlockSet, RemoteBlocks, SerializedNixlBlockSet};
use anyhow::{Context, Result};

use super::*;

pub fn create_lw_sharded_factories(block_sets: Vec<SerializedNixlBlockSet>) -> Result<()> {
    // deserialize to block sets
    let count = block_sets.len();

    let block_sets = block_sets
        .into_iter()
        .map(|bs| NixlBlockSet::try_from(bs).context("Failed to deserialize remote blockset"))
        .collect::<Result<Vec<NixlBlockSet>>>()?;

    let mut worker_ids = Vec::with_capacity(count);
    let mut nixl_metadata = Vec::with_capacity(count);
    let mut disk_blocks = HashMap::with_capacity(count);
    let mut host_blocks = HashMap::with_capacity(count);
    let mut device_blocks = HashMap::with_capacity(count);

    for block_set in block_sets {
        let (block_sets, metadata, worker_id) = block_set.dissolve();

        worker_ids.push(worker_id);
        nixl_metadata.push(metadata);

        for (block_set_idx, block_set_layout) in block_sets {
            let remote_blocks =
                RemoteBlocks::from_serialized(block_set_layout.clone(), block_set_idx, worker_id)?;

            match remote_blocks.layout().storage_type() {
                StorageType::Disk(_) => {
                    disk_blocks.insert(worker_id, remote_blocks);
                }
                StorageType::Pinned => {
                    host_blocks.insert(worker_id, remote_blocks);
                }
                StorageType::Device(_device_id) => {
                    device_blocks.insert(worker_id, remote_blocks);
                }
                _ => {
                    anyhow::bail!(
                        "Unsupported storage type: {:?}",
                        // TODO(ryan): implement Display for StorageType
                        remote_blocks.layout().storage_type()
                    );
                }
            }
        }

        // for each worker, check that the configuration and block counts match across
        // worker_ids; use worker[0] as the reference
        let reference_worker_id = worker_ids[0];
        let reference_disk_blocks = disk_blocks.get(&reference_worker_id);
        let reference_host_blocks = host_blocks.get(&reference_worker_id);
        let reference_device_blocks = device_blocks.get(&reference_worker_id);

        for worker_id in worker_ids.iter().skip(1) {
            let disk_blocks = disk_blocks.get(worker_id);
            let host_blocks = host_blocks.get(worker_id);
            let device_blocks = device_blocks.get(worker_id);

            validate_block_compatibility(reference_disk_blocks, disk_blocks).with_context(
                || {
                    format!(
                        "Worker {} has incompatible disk blocks with worker {}",
                        reference_worker_id, worker_id
                    )
                },
            )?;

            validate_block_compatibility(reference_host_blocks, host_blocks).with_context(
                || {
                    format!(
                        "Worker {} has incompatible host blocks with worker {}",
                        reference_worker_id, worker_id
                    )
                },
            )?;

            validate_block_compatibility(reference_device_blocks, device_blocks).with_context(
                || {
                    format!(
                        "Worker {} has incompatible device blocks with worker {}",
                        reference_worker_id, worker_id
                    )
                },
            )?;
        }
    }

    unimplemented!()
}

fn validate_block_compatibility(
    reference_blocks: Option<&RemoteBlocks>,
    other_blocks: Option<&RemoteBlocks>,
) -> Result<()> {
    if reference_blocks.is_none() && other_blocks.is_none() {
        return Ok(());
    }

    if reference_blocks.is_none() || other_blocks.is_none() {
        anyhow::bail!("One of the blocks is None");
    }

    let reference_layout = reference_blocks.unwrap().layout();
    let other_layout = other_blocks.unwrap().layout();

    if reference_layout.storage_type() != other_layout.storage_type() {
        anyhow::bail!("Storage types do not match");
    }

    if reference_layout.config() != other_layout.config() {
        anyhow::bail!("Layout configurations do not match");
    }

    Ok(())
}
