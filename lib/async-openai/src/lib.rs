// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Based on https://github.com/64bit/async-openai/ by Himanshu Neema
// Original Copyright (c) 2022 Himanshu Neema
// Licensed under MIT License (see ATTRIBUTIONS-Rust.md)
//
// Modifications Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
// Licensed under Apache 2.0

//! ## Examples
//! For full working examples for all supported features see [examples](https://github.com/64bit/async-openai/tree/main/examples) directory in the repository.
//!
#![allow(deprecated)]
#![allow(warnings)]
#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(feature = "byot")]
pub(crate) use async_openai_macros::byot;

#[cfg(not(feature = "byot"))]
pub(crate) use async_openai_macros::byot_passthrough as byot;

mod assistants;
mod audio;
mod audit_logs;
mod batches;
mod chat;
mod client;
mod completion;
pub mod config;
mod download;
mod embedding;
pub mod error;
mod file;
mod fine_tuning;
mod image;
mod invites;
mod messages;
mod model;
mod moderation;
mod project_api_keys;
mod project_service_accounts;
mod project_users;
mod projects;
mod responses;
mod runs;
mod steps;
mod threads;
pub mod traits;
pub mod types;
mod uploads;
mod users;
mod util;
mod vector_store_file_batches;
mod vector_store_files;
mod vector_stores;

pub use assistants::Assistants;
pub use audio::Audio;
pub use audit_logs::AuditLogs;
pub use batches::Batches;
pub use chat::Chat;
pub use client::Client;
pub use completion::Completions;
pub use embedding::Embeddings;
pub use file::Files;
pub use fine_tuning::FineTuning;
pub use image::Images;
pub use invites::Invites;
pub use messages::Messages;
pub use model::Models;
pub use moderation::Moderations;
pub use project_api_keys::ProjectAPIKeys;
pub use project_service_accounts::ProjectServiceAccounts;
pub use project_users::ProjectUsers;
pub use projects::Projects;
pub use responses::Responses;
pub use runs::Runs;
pub use steps::Steps;
pub use threads::Threads;
pub use uploads::Uploads;
pub use users::Users;
pub use vector_store_file_batches::VectorStoreFileBatches;
pub use vector_store_files::VectorStoreFiles;
pub use vector_stores::VectorStores;
