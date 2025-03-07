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

use futures_util::TryStreamExt;
use netlink_packet_route::address::AddressAttribute;
use netlink_packet_route::link::LinkLayerType;
use netlink_packet_route::link::State as LinkState;
use netlink_packet_route::link::{LinkAttribute, LinkMessage};
use netlink_packet_route::AddressFamily;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::{collections::HashMap, error::Error};

pub async fn get_primary_interface() -> Result<Option<String>, LinkDataError> {
    let mut candidates: VecDeque<String> = get_ipv4_interface_links()
        .await?
        .into_iter()
        .filter(|(k, v)| v.is_ethernet() && v.link_is_up() && v.has_carrier() && k.starts_with("e"))
        .map(|(k, _)| k)
        .collect();

    Ok(candidates.pop_front())
}

#[derive(Clone, Debug)]
// Most of the fields are Option<T> because the netlink protocol allows them
// to be absent (even though we have no reason to believe they'd ever actually
// be missing).
struct InterfaceLinkData {
    link_type: LinkLayerType,
    state: Option<LinkState>,
    has_carrier: bool,
}

impl InterfaceLinkData {
    pub fn link_is_up(&self) -> bool {
        self.state
            .map(|state| matches!(state, LinkState::Up))
            .unwrap_or(false)
    }

    pub fn is_ethernet(&self) -> bool {
        matches!(self.link_type, LinkLayerType::Ether)
    }

    pub fn has_carrier(&self) -> bool {
        self.has_carrier
    }
}

impl From<LinkMessage> for InterfaceLinkData {
    fn from(link_message: LinkMessage) -> Self {
        let link_type = link_message.header.link_layer_type;
        let state = link_message
            .attributes
            .iter()
            .find_map(|attribute| match attribute {
                LinkAttribute::OperState(state) => Some(*state),
                _ => None,
            });
        let has_carrier = link_message
            .attributes
            .iter()
            .find_map(|attribute| match attribute {
                LinkAttribute::Carrier(1) => Some(true),
                _ => None,
            })
            .unwrap_or(false);
        InterfaceLinkData {
            link_type,
            state,
            has_carrier,
        }
    }
}

#[derive(Debug)]
pub struct LinkDataError {
    kind: LinkDataErrorKind,
    interface: Option<String>,
}

impl LinkDataError {
    fn connection(connection_error: std::io::Error) -> Self {
        let kind = LinkDataErrorKind::Connection(connection_error);
        let interface = None;
        Self { kind, interface }
    }

    fn communication(communication_error: rtnetlink::Error) -> Self {
        let kind = LinkDataErrorKind::Communication(communication_error);
        let interface = None;
        Self { kind, interface }
    }
}

impl std::fmt::Display for LinkDataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let err_message = "could not get interface link data";
        if let Some(interface) = self.interface.as_ref() {
            write!(f, "{err_message} for {interface}")
        } else {
            write!(f, "{err_message}")
        }
    }
}

impl Error for LinkDataError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self.kind {
            LinkDataErrorKind::Connection(ref e) => Some(e),
            LinkDataErrorKind::Communication(ref e) => Some(e),
        }
    }
}

#[derive(Debug)]
pub enum LinkDataErrorKind {
    Connection(std::io::Error),
    Communication(rtnetlink::Error),
}

// Retrieve the link data (state, MTU, etc.) for all interfaces, and return
// them as a HashMap keyed by interface name. This is roughly equivalent to `ip
// link show` since we're using the same netlink interface under the hood as
// that command.
async fn get_ipv4_interface_links() -> Result<HashMap<String, InterfaceLinkData>, LinkDataError> {
    let (netlink_connection, rtnetlink_handle, _receiver) =
        rtnetlink::new_connection().map_err(LinkDataError::connection)?;

    // We have to spawn off the netlink connection because of the architecture
    // of `netlink_proto::Connection`, which runs in the background and owns
    // the socket. We communicate with it via channel messages, and it will exit
    // when both `rtnetlink_handle` and `_receiver` go out of scope.
    tokio::spawn(netlink_connection);

    let address_handle = rtnetlink_handle.address().get().execute();
    let ipv4s: HashSet<String> = address_handle
        .try_filter_map(|addr_message| async move {
            if matches!(addr_message.header.family, AddressFamily::Inet) {
                Ok(addr_message
                    .attributes
                    .into_iter()
                    .find(|attr| matches!(attr, AddressAttribute::Label(_)))
                    .and_then(|x| match x {
                        AddressAttribute::Label(label) => Some(label),
                        _ => None,
                    }))
            } else {
                Ok(None)
            }
        })
        .try_collect()
        .await
        .map_err(LinkDataError::communication)?;

    let link_handle = rtnetlink_handle.link().get().execute();
    link_handle
        .try_filter_map(|link_message| async {
            let maybe_interface_data = match extract_interface_name(&link_message) {
                Some(interface_name) => {
                    if ipv4s.contains(&interface_name) {
                        Some((interface_name, InterfaceLinkData::from(link_message)))
                    } else {
                        None
                    }
                }
                None => {
                    let idx = link_message.header.index;
                    eprintln!(
                        "Network interface with index {idx} doesn't have a name (no IfName attribute)"
                    );
                    None
                }
            };
            Ok(maybe_interface_data)
        })
        .try_collect()
        .await
        .map_err(LinkDataError::communication)
}

fn extract_interface_name(link_message: &LinkMessage) -> Option<String> {
    link_message
        .attributes
        .iter()
        .find_map(|attribute| match attribute {
            LinkAttribute::IfName(name) => Some(name.clone()),
            _ => None,
        })
}
