// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::*;

use anyhow::Result;
use nixl_sys::{MemoryRegion, NixlDescriptor, XferDescList};
use std::collections::HashMap;
use std::future::Future;

type DeviceId = u64;
type LayerDim = usize;
type OuterDim = usize;
type Addr = usize;
type Size = usize;

#[derive(Debug)]
struct XferAggr {
    descriptors: HashMap<(DeviceId, LayerDim, OuterDim), Vec<(Addr, Size)>>,
}

impl XferAggr {
    fn new() -> Self {
        Self {
            descriptors: HashMap::new(),
        }
    }

    fn can_extend(
        &self,
        addr: Addr,
        device_id: DeviceId,
        layer_dim: LayerDim,
        outer_dim: OuterDim,
    ) -> bool {
        if let Some(device_descs) = self.descriptors.get(&(device_id, layer_dim, outer_dim)) {
            if let Some(last) = device_descs.last() {
                return last.0 + last.1 == addr;
            }
        }
        false
    }

    fn add_desc(
        &mut self,
        addr: Addr,
        size: Size,
        device_id: DeviceId,
        layer_dim: LayerDim,
        outer_dim: OuterDim,
        extend: bool,
    ) {
        let device_descs = self
            .descriptors
            .entry((device_id, layer_dim, outer_dim))
            .or_insert_with(Vec::new);

        if extend && let Some(last) = device_descs.last_mut() {
            last.1 += size;
        } else {
            device_descs.push((addr, size));
        }
    }

    fn populate_xfer_desc_list(&self, xfer_desc_list: &mut XferDescList) -> Result<()> {
        for (&(device_id, layer_dim, outer_dim), descs) in &self.descriptors {
            for &(addr, size) in descs {
                tracing::info!(
                    "tid {}<{:x}, {}, {}, {}, {}>",
                    nix::unistd::gettid(),
                    addr,
                    size,
                    device_id,
                    layer_dim,
                    outer_dim
                );
                xfer_desc_list.add_desc(addr, size, device_id)?;
            }
        }

        Ok(())
    }
}

fn append_xfer_request<Source, Destination>(
    src: &Source,
    dst: &mut Destination,
    src_aggr: &mut XferAggr,
    dst_aggr: &mut XferAggr,
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
            let src_addr = src_desc.as_ptr() as usize;
            let src_size = src_desc.size();
            let src_device_id = src_desc.device_id();

            let dst_addr = dst_desc.as_ptr() as usize;
            let dst_size = dst_desc.size();
            let dst_device_id = dst_desc.device_id();

            // Check if both can be extended (use outer_dim = 0 for fully contiguous case)
            let src_can_extend = src_aggr.can_extend(src_addr, src_device_id, 0, 0);
            let dst_can_extend = dst_aggr.can_extend(dst_addr, dst_device_id, 0, 0);
            let extend_both = src_can_extend && dst_can_extend;

            src_aggr.add_desc(src_addr, src_size, src_device_id, 0, 0, extend_both);
            dst_aggr.add_desc(dst_addr, dst_size, dst_device_id, 0, 0, extend_both);
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
                    let src_addr = src_desc.as_ptr() as usize;
                    let src_size = src_desc.size();
                    let src_device_id = src_desc.device_id();

                    let dst_addr = dst_desc.as_ptr() as usize;
                    let dst_size = dst_desc.size();
                    let dst_device_id = dst_desc.device_id();

                    // Check if both can be extended (use actual outer_idx for layered case)
                    let src_can_extend =
                        src_aggr.can_extend(src_addr, src_device_id, layer_idx, outer_idx);
                    let dst_can_extend =
                        dst_aggr.can_extend(dst_addr, dst_device_id, layer_idx, outer_idx);
                    let extend_both = src_can_extend && dst_can_extend;

                    src_aggr.add_desc(
                        src_addr,
                        src_size,
                        src_device_id,
                        layer_idx,
                        outer_idx,
                        extend_both,
                    );
                    dst_aggr.add_desc(
                        dst_addr,
                        dst_size,
                        dst_device_id,
                        layer_idx,
                        outer_idx,
                        extend_both,
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

    let mut src_aggr = XferAggr::new();
    let mut dst_aggr = XferAggr::new();

    for (src, dst) in src.iter().zip(dst.iter_mut()) {
        append_xfer_request(src, dst, &mut src_aggr, &mut dst_aggr)?;
    }

    let mut src_dl = XferDescList::new(src_mem_type, false)?;
    let mut dst_dl = XferDescList::new(dst_mem_type, false)?;

    src_aggr.populate_xfer_desc_list(&mut src_dl)?;
    dst_aggr.populate_xfer_desc_list(&mut dst_dl)?;

    debug_assert!(!src_dl.has_overlaps()? && !dst_dl.has_overlaps()?);

    let xfer_req = nixl_agent.create_xfer_req(
        transfer_type.as_xfer_op(),
        &src_dl,
        &dst_dl,
        &nixl_agent.name(),
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
                    Ok(false) => break, // Transfer is complete.
                    Ok(true) => tokio::time::sleep(std::time::Duration::from_millis(5)).await, // Transfer is still in progress.
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
