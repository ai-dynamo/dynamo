// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use anyhow::Result;
use nixl_sys::{MemoryRegion, NixlDescriptor, XferDescList, XferStatus};
use std::future::Future;

fn append_xfer_request<Source, Destination>(
    src: &Source,
    dst: &mut Destination,
    src_dl: &mut XferDescList,
    dst_dl: &mut XferDescList,
) -> Result<()>
where
    Source: BlockDataProvider,
    Source::StorageType: NixlDescriptor,
    Destination: BlockDataProviderMut,
    Destination::StorageType: NixlDescriptor,
{
    let src_data = src.block_data();
    let dst_data = dst.block_data_mut();

    if src_data.is_fully_contiguous() && dst_data.is_fully_contiguous() {
        let src_desc = src_data.block_view()?.as_nixl_descriptor();
        let dst_desc = dst_data.block_view_mut()?.as_nixl_descriptor_mut();

        unsafe {
            src_dl.add_desc(
                src_desc.as_ptr() as usize,
                src_desc.size(),
                src_desc.device_id(),
            );

            dst_dl.add_desc(
                dst_desc.as_ptr() as usize,
                dst_desc.size(),
                dst_desc.device_id(),
            );
        }

        Ok(())
    } else {
        assert_eq!(src_data.num_layers(), dst_data.num_layers());
        for layer_idx in 0..src_data.num_layers() {
            for outer_idx in 0..src_data.num_outer_dims() {
                let src_view = src_data.layer_view(layer_idx, outer_idx)?;
                let mut dst_view = dst_data.layer_view_mut(layer_idx, outer_idx)?;

                debug_assert_eq!(src_view.size(), dst_view.size());

                let src_desc = src_view.as_nixl_descriptor();
                let dst_desc = dst_view.as_nixl_descriptor_mut();

                unsafe {
                    src_dl.add_desc(
                        src_desc.as_ptr() as usize,
                        src_desc.size(),
                        src_desc.device_id(),
                    );

                    dst_dl.add_desc(
                        dst_desc.as_ptr() as usize,
                        dst_desc.size(),
                        dst_desc.device_id(),
                    );
                }
            }
        }
        Ok(())
    }
}

fn append_read_xfer_request<RemoteSource, LocalDestination>(
    remote_src: &RemoteSource,
    local_dst: &mut LocalDestination,
    local_dl: &mut XferDescList,
    remote_dl: &mut XferDescList,
) -> Result<()>
where
    RemoteSource: BlockDataProvider,
    RemoteSource::StorageType: NixlDescriptor,
    LocalDestination: BlockDataProviderMut,
    LocalDestination::StorageType: NixlDescriptor,
{
    let remote_data = remote_src.block_data();
    let local_data = local_dst.block_data_mut();

    if remote_data.is_fully_contiguous() && local_data.is_fully_contiguous() {
        let remote_desc = remote_data.block_view()?.as_nixl_descriptor();
        let local_desc = local_data.block_view_mut()?.as_nixl_descriptor_mut();

        unsafe {
            local_dl.add_desc(
                local_desc.as_ptr() as usize,
                local_desc.size(),
                local_desc.device_id(),
            );

            remote_dl.add_desc(
                remote_desc.as_ptr() as usize,
                remote_desc.size(),
                remote_desc.device_id(),
            );
        }

        Ok(())
    } else {
        assert_eq!(remote_data.num_layers(), local_data.num_layers());
        for layer_idx in 0..remote_data.num_layers() {
            for outer_idx in 0..remote_data.num_outer_dims() {
                let remote_view = remote_data.layer_view(layer_idx, outer_idx)?;
                let mut local_view = local_data.layer_view_mut(layer_idx, outer_idx)?;

                debug_assert_eq!(remote_view.size(), local_view.size());

                let remote_desc = remote_view.as_nixl_descriptor();
                let local_desc = local_view.as_nixl_descriptor_mut();

                unsafe {
                    local_dl.add_desc(
                        local_desc.as_ptr() as usize,
                        local_desc.size(),
                        local_desc.device_id(),
                    );

                    remote_dl.add_desc(
                        remote_desc.as_ptr() as usize,
                        remote_desc.size(),
                        remote_desc.device_id(),
                    );
                }
            }
        }
        Ok(())
    }
}

/// Copy a block from a source to a destination using CUDA memcpy
pub fn write_blocks_to<Source, Destination>(
    src: &[Source],
    dst: &mut [Destination],
    ctx: &Arc<TransferContext>,
    transfer_type: NixlTransfer,
) -> Result<Box<dyn Future<Output = ()> + Send + Sync + Unpin>>
where
    Source: BlockDataProvider,
    Source::StorageType: NixlDescriptor,
    Destination: BlockDataProviderMut,
    Destination::StorageType: NixlDescriptor,
{
    if src.is_empty() || dst.is_empty() {
        return Ok(Box::new(std::future::ready(())));
    }
    assert_eq!(src.len(), dst.len());

    let nixl_agent_arc = ctx.as_ref().nixl_agent();
    let nixl_agent = nixl_agent_arc
        .as_ref()
        .as_ref()
        .expect("NIXL agent not found");

    let src_mem_type = src
        .first()
        .unwrap()
        .block_data()
        .storage_type()
        .nixl_mem_type();
    let dst_mem_type = dst
        .first()
        .unwrap()
        .block_data()
        .storage_type()
        .nixl_mem_type();

    let mut local_dl = match transfer_type {
        NixlTransfer::Read => XferDescList::new(dst_mem_type)?,
        NixlTransfer::Write => XferDescList::new(src_mem_type)?,
    };
    let mut remote_dl = match transfer_type {
        NixlTransfer::Read => XferDescList::new(src_mem_type)?,
        NixlTransfer::Write => XferDescList::new(dst_mem_type)?,
    };

    for (src, dst) in src.iter().zip(dst.iter_mut()) {
        match transfer_type {
            NixlTransfer::Read => {
                append_read_xfer_request(src, dst, &mut local_dl, &mut remote_dl)?
            }
            NixlTransfer::Write => append_xfer_request(src, dst, &mut local_dl, &mut remote_dl)?,
        }
    }

    let local_agent_name = nixl_agent.name();
    let remote_worker_id = match transfer_type {
        NixlTransfer::Read => src
            .first()
            .expect("non-empty remote source list")
            .block_data()
            .worker_id(),
        NixlTransfer::Write => dst
            .first()
            .expect("non-empty remote destination list")
            .block_data()
            .worker_id(),
    };
    let remote_agent_name = if remote_worker_id.to_string() == local_agent_name {
        local_agent_name.clone()
    } else {
        remote_worker_id.to_string()
    };

    let xfer_req = nixl_agent.create_xfer_req(
        transfer_type.as_xfer_op(),
        &local_dl,
        &remote_dl,
        &remote_agent_name,
        None,
    )?;

    let still_pending = nixl_agent.post_xfer_req(&xfer_req, None)?;

    if still_pending {
        Ok(Box::new(Box::pin(async move {
            let nixl_agent = nixl_agent_arc
                .as_ref()
                .as_ref()
                .expect("NIXL agent not found");

            loop {
                match nixl_agent.get_xfer_status(&xfer_req) {
                    Ok(XferStatus::Success) => break, // Transfer is complete.
                    Ok(XferStatus::InProgress) => {
                        tokio::time::sleep(std::time::Duration::from_millis(5)).await
                    } // Transfer is still in progress.
                    Err(e) => {
                        tracing::error!("Error getting transfer status: {}", e);
                        break;
                    }
                }
            }
        })))
    } else {
        Ok(Box::new(std::future::ready(())))
    }
}
