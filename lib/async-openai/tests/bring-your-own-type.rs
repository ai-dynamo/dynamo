// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Based on https://github.com/64bit/async-openai/ by Himanshu Neema
// Original Copyright (c) 2022 Himanshu Neema
// Licensed under MIT License (see ATTRIBUTIONS-Rust.md)
//
// Modifications Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
// Licensed under Apache 2.0

#![allow(dead_code)]
//! The purpose of this test to make sure that all _byot methods compiles with custom types.
use std::pin::Pin;

use dynamo_async_openai::{Client, error::OpenAIError};
use futures::Stream;
use serde_json::{Value, json};

impl dynamo_async_openai::traits::AsyncTryFrom<MyJson> for reqwest::multipart::Form {
    type Error = OpenAIError;
    async fn try_from(_value: MyJson) -> Result<Self, Self::Error> {
        Ok(reqwest::multipart::Form::new())
    }
}

#[derive(Clone)]
pub struct MyJson(Value);

type MyStreamingType = Pin<Box<dyn Stream<Item = Result<Value, OpenAIError>> + Send>>;

#[tokio::test]
async fn test_byot_files() {
    let client = Client::new();

    let _r: Result<Value, OpenAIError> = client.files().create_byot(MyJson(json!({}))).await;
    let _r: Result<Value, OpenAIError> = client.files().list_byot([("limit", "2")]).await;
    let _r: Result<Value, OpenAIError> = client.files().retrieve_byot("file_id").await;
    let _r: Result<Value, OpenAIError> = client.files().delete_byot("file_id").await;
}

#[tokio::test]
async fn test_byot_assistants() {
    let client = Client::new();

    let _r: Result<Value, OpenAIError> = client.assistants().create_byot(json!({})).await;
    let _r: Result<Value, OpenAIError> = client.assistants().retrieve_byot("aid").await;
    let _r: Result<Value, OpenAIError> = client.assistants().update_byot("aid", json!({})).await;
    let _r: Result<Value, OpenAIError> = client.assistants().list_byot([("limit", 2)]).await;
}

#[tokio::test]
async fn test_byot_models() {
    let client = Client::new();

    let _r: Result<Value, OpenAIError> = client.models().list_byot().await;
    let _r: Result<Value, OpenAIError> = client.models().retrieve_byot("").await;
    let _r: Result<Value, OpenAIError> = client.models().delete_byot(String::new()).await;
}

#[tokio::test]
async fn test_byot_moderations() {
    let client = Client::new();

    let _r: Result<Value, OpenAIError> = client.moderations().create_byot(json!({})).await;
}

#[tokio::test]
async fn test_byot_images() {
    let client = Client::new();

    let _r: Result<Value, OpenAIError> = client.images().create_byot(json!({})).await;
    let _r: Result<Value, OpenAIError> = client.images().create_edit_byot(MyJson(json!({}))).await;
    let _r: Result<Value, OpenAIError> = client
        .images()
        .create_variation_byot(MyJson(json!({})))
        .await;
}

#[tokio::test]
async fn test_byot_chat() {
    let client = Client::new();

    let _r: Result<Value, OpenAIError> = client.chat().create_byot(json!({})).await;
    let _r: Result<MyStreamingType, OpenAIError> =
        client.chat().create_stream_byot(json!({})).await;
}

#[tokio::test]
async fn test_byot_completions() {
    let client = Client::new();

    let _r: Result<Value, OpenAIError> = client.completions().create_byot(json!({})).await;
    let _r: Result<MyStreamingType, OpenAIError> =
        client.completions().create_stream_byot(json!({})).await;
}

#[tokio::test]
async fn test_byot_audio() {
    let client = Client::new();

    let _r: Result<Value, OpenAIError> = client.audio().transcribe_byot(MyJson(json!({}))).await;
    let _r: Result<Value, OpenAIError> = client
        .audio()
        .transcribe_verbose_json_byot(MyJson(json!({})))
        .await;
    let _r: Result<Value, OpenAIError> = client.audio().translate_byot(MyJson(json!({}))).await;
    let _r: Result<Value, OpenAIError> = client
        .audio()
        .translate_verbose_json_byot(MyJson(json!({})))
        .await;
}

#[tokio::test]
async fn test_byot_embeddings() {
    let client = Client::new();

    let _r: Result<Value, OpenAIError> = client.embeddings().create_byot(json!({})).await;
    let _r: Result<Value, OpenAIError> = client.embeddings().create_base64_byot(json!({})).await;
}

#[tokio::test]
async fn test_byot_fine_tunning() {
    let client = Client::new();

    let _r: Result<Value, OpenAIError> = client.fine_tuning().create_byot(json!({})).await;
    let _r: Result<Value, OpenAIError> = client
        .fine_tuning()
        .list_paginated_byot([("limit", "2")])
        .await;
    let _r: Result<Value, OpenAIError> = client
        .fine_tuning()
        .retrieve_byot("fine_tunning_job_id")
        .await;
    let _r: Result<Value, OpenAIError> =
        client.fine_tuning().cancel_byot("fine_tuning_job_id").await;
    let _r: Result<Value, OpenAIError> = client
        .fine_tuning()
        .list_events_byot("fine_tuning_job_id", [("limit", "2")])
        .await;
    let _r: Result<Value, OpenAIError> = client
        .fine_tuning()
        .list_checkpoints_byot("fine_tuning_job_id", [("limit", "2")])
        .await;
}

#[derive(Clone, serde::Deserialize)]
pub struct MyThreadJson(Value);

impl TryFrom<eventsource_stream::Event> for MyThreadJson {
    type Error = OpenAIError;
    fn try_from(_value: eventsource_stream::Event) -> Result<Self, Self::Error> {
        Ok(MyThreadJson(json!({})))
    }
}

type MyThreadStreamingType = Pin<Box<dyn Stream<Item = Result<MyThreadJson, OpenAIError>> + Send>>;

#[tokio::test]
async fn test_byot_threads() {
    let client = Client::new();

    let _r: Result<Value, OpenAIError> = client.threads().create_and_run_byot(json!({})).await;
    let _r: Result<MyThreadStreamingType, OpenAIError> =
        client.threads().create_and_run_stream_byot(json!({})).await;
    let _r: Result<Value, OpenAIError> = client.threads().create_byot(json!({})).await;
    let _r: Result<Value, OpenAIError> = client.threads().retrieve_byot("thread_id").await;
    let _r: Result<Value, OpenAIError> = client.threads().update_byot("thread_id", json!({})).await;
    let _r: Result<Value, OpenAIError> = client.threads().delete_byot("thread_id").await;
}

#[tokio::test]
async fn test_byot_messages() {
    let client = Client::new();

    let _r: Result<Value, OpenAIError> = client
        .threads()
        .messages("thread_id")
        .create_byot(json!({}))
        .await;
    let _r: Result<Value, OpenAIError> = client
        .threads()
        .messages("thread_id")
        .retrieve_byot("message_id")
        .await;
    let _r: Result<Value, OpenAIError> = client
        .threads()
        .messages("thread_id")
        .update_byot("message_id", json!({}))
        .await;
    let _r: Result<Value, OpenAIError> = client
        .threads()
        .messages("thread_id")
        .list_byot([("limit", "2")])
        .await;
    let _r: Result<Value, OpenAIError> = client
        .threads()
        .messages("thread_id")
        .delete_byot("message_id")
        .await;
}

#[tokio::test]
async fn test_byot_runs() {
    let client = Client::new();

    let _r: Result<Value, OpenAIError> = client
        .threads()
        .runs("thread_id")
        .create_byot(json!({}))
        .await;
    let _r: Result<MyThreadStreamingType, OpenAIError> = client
        .threads()
        .runs("thread_id")
        .create_stream_byot(json!({}))
        .await;
    let _r: Result<Value, OpenAIError> = client
        .threads()
        .runs("thread_id")
        .retrieve_byot("run_id")
        .await;
    let _r: Result<Value, OpenAIError> = client
        .threads()
        .runs("thread_id")
        .update_byot("run_id", json!({}))
        .await;
    let _r: Result<Value, OpenAIError> = client
        .threads()
        .runs("thread_id")
        .list_byot([("limit", "2")])
        .await;
    let _r: Result<Value, OpenAIError> = client
        .threads()
        .runs("thread_id")
        .submit_tool_outputs_byot("run_id", json!({}))
        .await;
    let _r: Result<MyThreadStreamingType, OpenAIError> = client
        .threads()
        .runs("thread_id")
        .submit_tool_outputs_stream_byot("run_id", json!({}))
        .await;
    let _r: Result<Value, OpenAIError> = client
        .threads()
        .runs("thread_id")
        .cancel_byot("run_id")
        .await;
}

#[tokio::test]
async fn test_byot_run_steps() {
    let client = Client::new();

    let _r: Result<Value, OpenAIError> = client
        .threads()
        .runs("thread_id")
        .steps("run_id")
        .retrieve_byot("step_id")
        .await;
    let _r: Result<Value, OpenAIError> = client
        .threads()
        .runs("thread_id")
        .steps("run_id")
        .list_byot([("limit", "2")])
        .await;
}

#[tokio::test]
async fn test_byot_vector_store_files() {
    let client = Client::new();
    let _r: Result<Value, OpenAIError> = client
        .vector_stores()
        .files("vector_store_id")
        .create_byot(json!({}))
        .await;
    let _r: Result<Value, OpenAIError> = client
        .vector_stores()
        .files("vector_store_id")
        .retrieve_byot("file_id")
        .await;
    let _r: Result<Value, OpenAIError> = client
        .vector_stores()
        .files("vector_store_id")
        .delete_byot("file_id")
        .await;
    let _r: Result<Value, OpenAIError> = client
        .vector_stores()
        .files("vector_store_id")
        .list_byot([("limit", "2")])
        .await;
}

#[tokio::test]
async fn test_byot_vector_store_file_batches() {
    let client = Client::new();
    let _r: Result<Value, OpenAIError> = client
        .vector_stores()
        .file_batches("vector_store_id")
        .create_byot(json!({}))
        .await;
    let _r: Result<Value, OpenAIError> = client
        .vector_stores()
        .file_batches("vector_store_id")
        .retrieve_byot("file_id")
        .await;
    let _r: Result<Value, OpenAIError> = client
        .vector_stores()
        .file_batches("vector_store_id")
        .cancel_byot("file_id")
        .await;
    let _r: Result<Value, OpenAIError> = client
        .vector_stores()
        .file_batches("vector_store_id")
        .list_byot("batch_id", [("limit", "2")])
        .await;
}

#[tokio::test]
async fn test_byot_batches() {
    let client = Client::new();
    let _r: Result<Value, OpenAIError> = client.batches().create_byot(json!({})).await;
    let _r: Result<Value, OpenAIError> = client.batches().list_byot([("limit", "2")]).await;
    let _r: Result<Value, OpenAIError> = client.batches().retrieve_byot("batch_id").await;
    let _r: Result<Value, OpenAIError> = client.batches().cancel_byot("batch_id").await;
}

#[tokio::test]
async fn test_byot_audit_logs() {
    let client = Client::new();
    let _r: Result<Value, OpenAIError> = client.audit_logs().get_byot([("limit", "2")]).await;
}

#[tokio::test]
async fn test_byot_invites() {
    let client = Client::new();
    let _r: Result<Value, OpenAIError> = client.invites().create_byot(json!({})).await;
    let _r: Result<Value, OpenAIError> = client.invites().retrieve_byot("invite_id").await;
    let _r: Result<Value, OpenAIError> = client.invites().delete_byot("invite_id").await;
    let _r: Result<Value, OpenAIError> = client.invites().list_byot([("limit", "2")]).await;
}

#[tokio::test]
async fn test_byot_projects() {
    let client = Client::new();

    let _r: Result<Value, OpenAIError> = client.projects().list_byot([("limit", "2")]).await;
    let _r: Result<Value, OpenAIError> = client.projects().create_byot(json!({})).await;
    let _r: Result<Value, OpenAIError> = client.projects().retrieve_byot("project_id").await;
    let _r: Result<Value, OpenAIError> =
        client.projects().modify_byot("project_id", json!({})).await;
    let _r: Result<Value, OpenAIError> = client.projects().archive_byot("project_id").await;
}

#[tokio::test]
async fn test_byot_project_api_keys() {
    let client = Client::new();

    let _r: Result<Value, OpenAIError> = client
        .projects()
        .api_keys("project_id")
        .list_byot([("query", "2")])
        .await;

    let _r: Result<Value, OpenAIError> = client
        .projects()
        .api_keys("project_id")
        .retrieve_byot("api_key")
        .await;

    let _r: Result<Value, OpenAIError> = client
        .projects()
        .api_keys("project_id")
        .delete_byot("api_key")
        .await;
}

#[tokio::test]
async fn test_byot_project_service_accounts() {
    let client = Client::new();

    let _r: Result<Value, OpenAIError> = client
        .projects()
        .service_accounts("project_id")
        .create_byot(json!({}))
        .await;

    let _r: Result<Value, OpenAIError> = client
        .projects()
        .service_accounts("project_id")
        .delete_byot("service_account_id")
        .await;

    let _r: Result<Value, OpenAIError> = client
        .projects()
        .service_accounts("project_id")
        .retrieve_byot("service_account_id")
        .await;

    let _r: Result<Value, OpenAIError> = client
        .projects()
        .service_accounts("project_id")
        .list_byot([("limit", "2")])
        .await;
}

#[tokio::test]
async fn test_byot_project_users() {
    let client = Client::new();

    let _r: Result<Value, OpenAIError> = client
        .projects()
        .users("project_id")
        .create_byot(json!({}))
        .await;
    let _r: Result<Value, OpenAIError> = client
        .projects()
        .users("project_id")
        .delete_byot("user_id")
        .await;

    let _r: Result<Value, OpenAIError> = client
        .projects()
        .users("project_id")
        .list_byot([("limit", "2")])
        .await;

    let _r: Result<Value, OpenAIError> = client
        .projects()
        .users("project_id")
        .retrieve_byot("user_id")
        .await;
}

#[tokio::test]
async fn test_byot_uploads() {
    let client = Client::new();

    let _r: Result<Value, OpenAIError> = client.uploads().create_byot(json!({})).await;
    let _r: Result<Value, OpenAIError> = client
        .uploads()
        .add_part_byot("upload_id", MyJson(json!({})))
        .await;
    let _r: Result<Value, OpenAIError> =
        client.uploads().complete_byot("upload_id", json!({})).await;
    let _r: Result<Value, OpenAIError> = client.uploads().cancel_byot("upload_id").await;
}

#[tokio::test]
async fn test_byot_users() {
    let client = Client::new();

    let _r: Result<Value, OpenAIError> = client.users().list_byot([("limit", "2")]).await;
    let _r: Result<Value, OpenAIError> = client.users().modify_byot("user_id", json!({})).await;
    let _r: Result<Value, OpenAIError> = client.users().retrieve_byot("user_id").await;
    let _r: Result<Value, OpenAIError> = client.users().delete_byot("user_id").await;
}
