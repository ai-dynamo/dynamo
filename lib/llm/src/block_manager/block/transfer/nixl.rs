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
use nixl_sys::{MemoryRegion, NixlDescriptor, OptArgs, XferDescList, XferOp};
use std::future::{poll_fn, Future};
use std::ops::Range;
use std::task::Poll;

/// Copy a block from a source to a destination using CUDA memcpy
pub fn write_block_to<'a, Source, Destination>(
    src: &'a Source,
    dst: &'a mut Destination,
    ctx: Arc<TransferContext>,
    notify: Option<String>,
) -> Result<Box<dyn Future<Output = ()> + Send + Sync + Unpin>>
where
    Source: BlockDataProvider,
    Destination: BlockDataProviderMut,
{
    let src_data = src.block_data(private::PrivateToken);
    let dst_data = dst.block_data_mut(private::PrivateToken);

    if src_data.is_fully_contiguous() && dst_data.is_fully_contiguous() {
        // Keep the arc to use in the returned future.
        let nixl_agent_arc = ctx.as_ref().nixl_agent();

        let nixl_agent = nixl_agent_arc
            .as_ref()
            .as_ref()
            .expect("NIXL agent not found");

        let mut src_dl = XferDescList::new(src_data.storage_type().nixl_mem_type())?;
        let mut dst_dl = XferDescList::new(dst_data.storage_type().nixl_mem_type())?;

        let src_desc = src_data.block_view()?.as_nixl_descriptor();
        let dst_desc = dst_data.block_view_mut()?.as_nixl_descriptor_mut();

        unsafe {
            src_dl.add_desc(
                src_desc.as_ptr() as usize,
                src_desc.size(),
                src_desc.device_id(),
            )?;

            dst_dl.add_desc(
                dst_desc.as_ptr() as usize,
                dst_desc.size(),
                dst_desc.device_id(),
            )?;
        }

        let xfer_req = nixl_agent
            .create_xfer_req(XferOp::Write, &src_dl, &dst_dl, &nixl_agent.name(), None)
            .unwrap();

        let mut xfer_args = OptArgs::new()?;

        if let Some(notify) = notify {
            xfer_args.set_has_notification(true)?;
            xfer_args.set_notification_message(notify.as_bytes())?;
        }

        let _ = nixl_agent.post_xfer_req(&xfer_req, Some(&xfer_args))?;

        // Return a future that completes when the transfer is complete.
        // TODO: How efficient is this? Can we do better?
        Ok(Box::new(poll_fn(move |_cx| {
            let nixl_agent = nixl_agent_arc
                .as_ref()
                .as_ref()
                .expect("NIXL agent not found");

            // The nixl agent returns true if the transfer is still in progress.
            if !nixl_agent.get_xfer_status(&xfer_req).unwrap() {
                Poll::Ready(())
            } else {
                Poll::Pending
            }
        })))
    } else {
        assert_eq!(src_data.num_layers(), dst_data.num_layers());
        write_layers_to(0..src_data.num_layers(), src, dst, ctx, notify)
    }
}

/// Copy a range of layers from a source to a destination using CUDA memcpy
pub fn write_layers_to<'a, Source, Destination>(
    layer_range: Range<usize>,
    src: &'a Source,
    dst: &'a mut Destination,
    ctx: Arc<TransferContext>,
    notify: Option<String>,
) -> Result<Box<dyn Future<Output = ()> + Send + Sync + Unpin>>
where
    Source: BlockDataProvider,
    Destination: BlockDataProviderMut,
{
    let src_data = src.block_data(private::PrivateToken);
    let dst_data = dst.block_data_mut(private::PrivateToken);

    let nixl_agent_arc = ctx.as_ref().nixl_agent();
    let nixl_agent = nixl_agent_arc
        .as_ref()
        .as_ref()
        .expect("NIXL agent not found");

    let remote_worker_id = dst_data.worker_id.to_string();
    let mut src_dl = XferDescList::new(src_data.storage_type().nixl_mem_type())?;
    let mut dst_dl = XferDescList::new(dst_data.storage_type().nixl_mem_type())?;

    // #[cfg(debug_assertions)]
    // {
    //     let expected_strategy = <<Source as BlockDataProvider>::StorageType as WriteToStrategy<
    //         Destination::StorageType,
    //     >>::write_to_strategy();
    //     assert_eq!(strategy, expected_strategy);
    // }

    for layer_idx in layer_range {
        let src_view = src_data.layer_view(layer_idx)?;
        let mut dst_view = dst_data.layer_view_mut(layer_idx)?;

        debug_assert_eq!(src_view.size(), dst_view.size());

        let src_desc = src_view.as_nixl_descriptor();
        let dst_desc = dst_view.as_nixl_descriptor_mut();

        unsafe {
            src_dl.add_desc(
                src_desc.as_ptr() as usize,
                src_desc.size(),
                src_desc.device_id(),
            )?;

            dst_dl.add_desc(
                dst_desc.as_ptr() as usize,
                dst_desc.size(),
                dst_desc.device_id(),
            )?;
        }
    }

    let mut xfer_args = OptArgs::new()?;

    if let Some(notify) = notify {
        xfer_args.set_has_notification(true)?;
        xfer_args.set_notification_message(notify.as_bytes())?;
    }

    let xfer_req = nixl_agent.create_xfer_req(
        XferOp::Write,
        &src_dl,
        &dst_dl,
        &remote_worker_id,
        Some(&xfer_args),
    )?;

    let _ = nixl_agent.post_xfer_req(&xfer_req, Some(&xfer_args))?;

    Ok(Box::new(poll_fn(move |_cx| {
        let nixl_agent = nixl_agent_arc
            .as_ref()
            .as_ref()
            .expect("NIXL agent not found");
        if !nixl_agent.get_xfer_status(&xfer_req).unwrap() {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    })))
}
