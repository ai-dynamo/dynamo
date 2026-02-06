// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Note: The EventPublisher and EventSubscriber trait impls have been removed.
// Use dynamo_runtime::transports::event_plane::{EventPublisher, EventSubscriber} instead:
//   - EventPublisher::for_component(&component, topic)
//   - EventSubscriber::for_component(&component, topic)
