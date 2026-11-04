// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The table the store needs: string keys `pk` (HASH) and `sk` (RANGE),
//! on-demand capacity, and TTL on `expires_at`. Production tables also want
//! point-in-time recovery and, across regions, a Global Table replica per
//! region; [`ensure_table`] sets up neither, so it suits development and
//! dynamodb-local.

use std::time::Duration;

use anyhow::{Context, bail};
use aws_sdk_dynamodb::Client;
use aws_sdk_dynamodb::config::Region;
use aws_sdk_dynamodb::types::{
    AttributeDefinition, BillingMode, KeySchemaElement, KeyType, ScalarAttributeType,
    TableDescription, TableStatus, TimeToLiveSpecification, TimeToLiveStatus,
};
use url::Url;

use super::keys;

/// How long [`ensure_table`] and [`describe`] wait for a new table to become
/// active. DynamoDB usually takes seconds; dynamodb-local, none.
const ACTIVE_WITHIN: Duration = Duration::from_secs(300);
/// The first wait between checks; each wait doubles, up to
/// [`MAX_CHECK_EVERY`].
const FIRST_CHECK_AFTER: Duration = Duration::from_millis(250);
const MAX_CHECK_EVERY: Duration = Duration::from_secs(5);

/// Creates `table` unless it exists, waits until it is active, and turns on
/// TTL on `expires_at` unless it is on. Needs `dynamodb:CreateTable` besides
/// `dynamodb:DescribeTable`, and for TTL `dynamodb:DescribeTimeToLive` and
/// `dynamodb:UpdateTimeToLive`. Replicas that start together may all call
/// it: one creates the table, and the rest wait for it.
pub async fn ensure_table(client: &Client, table: &str) -> anyhow::Result<()> {
    if read(client, table).await?.is_none() {
        let created = client
            .create_table()
            .table_name(table)
            .attribute_definitions(attribute(keys::PK)?)
            .attribute_definitions(attribute(keys::SK)?)
            .key_schema(key(keys::PK, KeyType::Hash)?)
            .key_schema(key(keys::SK, KeyType::Range)?)
            .billing_mode(BillingMode::PayPerRequest)
            .send()
            .await;
        match created {
            Ok(_) => tracing::info!(table, "created the DynamoDB table for registrations"),
            // Another replica created it first.
            Err(error)
                if error
                    .as_service_error()
                    .is_some_and(|error| error.is_resource_in_use_exception()) => {}
            Err(error) => {
                return Err(error).with_context(|| format!("create DynamoDB table {table}"));
            }
        }
    }
    wait_until_active(client, table).await?;
    // TTL only collects garbage, so a TTL that cannot be checked or turned
    // on, as without the TTL permissions, is no reason to stop.
    if let Err(error) = ensure_ttl(client, table).await {
        tracing::warn!(
            table,
            error = %format!("{error:#}"),
            "cannot make sure that TTL is on for the DynamoDB table; until it is on \
             expires_at, items of retired pools stay in the table"
        );
    }
    Ok(())
}

/// Turns on TTL on `expires_at` unless it is on or being turned on. TTL is
/// checked on every run, not only the one that creates the table: a run that
/// stops before turning it on would otherwise leave it off, and every Query
/// would keep reading retired pools' items.
async fn ensure_ttl(client: &Client, table: &str) -> anyhow::Result<()> {
    if ttl_is_on(client, table).await? {
        return Ok(());
    }
    let turned_on = client
        .update_time_to_live()
        .table_name(table)
        .time_to_live_specification(
            TimeToLiveSpecification::builder()
                .attribute_name(keys::EXPIRES_AT)
                .enabled(true)
                .build()?,
        )
        .send()
        .await;
    match turned_on {
        Ok(_) => {
            tracing::info!(
                table,
                "turned on TTL for the DynamoDB table for registrations"
            );
            Ok(())
        }
        // Another replica may have turned it on meanwhile, and DynamoDB
        // refuses a second TTL change for a while.
        Err(error) => match ttl_is_on(client, table).await {
            Ok(true) => Ok(()),
            _ => Err(error).with_context(|| format!("turn on TTL for DynamoDB table {table}")),
        },
    }
}

/// Whether TTL is on `expires_at`, or being turned on. TTL on another
/// attribute is an error: a table has one TTL attribute, and items would
/// never expire.
async fn ttl_is_on(client: &Client, table: &str) -> anyhow::Result<bool> {
    let output = client
        .describe_time_to_live()
        .table_name(table)
        .send()
        .await
        .with_context(|| format!("describe the TTL of DynamoDB table {table}"))?;
    let Some(ttl) = output.time_to_live_description() else {
        return Ok(false);
    };
    match ttl.time_to_live_status() {
        Some(TimeToLiveStatus::Enabled | TimeToLiveStatus::Enabling) => {
            if ttl.attribute_name() != Some(keys::EXPIRES_AT) {
                bail!(
                    "DynamoDB table {table} has TTL on {}, but registrations expire on {}",
                    ttl.attribute_name().unwrap_or("another attribute"),
                    keys::EXPIRES_AT
                );
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}

/// Checks that `table` exists with the keys the store writes, and waits for
/// it if it is still being created. The error for a missing table says how
/// to create it where the service looks for it: in `client`'s region, and
/// through `endpoint` if the service uses one.
pub async fn describe(client: &Client, table: &str, endpoint: Option<&Url>) -> anyhow::Result<()> {
    if read(client, table).await?.is_none() {
        let options = cli_options(client.config().region(), endpoint);
        bail!(
            "DynamoDB table {table} does not exist. Create it with\n  {}\nand\n  {}\nor start \
             with --dynamodb-ensure-table",
            create_table_command(table, &options),
            update_time_to_live_command(table, &options),
        );
    }
    let description = wait_until_active(client, table).await?;
    check_keys(table, &description)
}

/// The `aws` CLI options that reach the table the service uses: its region,
/// and its endpoint when that is not the region's own, as for
/// dynamodb-local. Without them, the CLI's own defaults could create the
/// table somewhere else.
pub fn cli_options(region: Option<&Region>, endpoint: Option<&Url>) -> String {
    let mut options = String::new();
    if let Some(region) = region {
        options.push_str(&format!(" --region {region}"));
    }
    if let Some(endpoint) = endpoint {
        let endpoint = endpoint.as_str().trim_end_matches('/');
        options.push_str(&format!(" --endpoint-url {endpoint}"));
    }
    options
}

/// `aws dynamodb create-table` for `table`, then `cli_options`.
pub fn create_table_command(table: &str, cli_options: &str) -> String {
    format!(
        "aws dynamodb create-table --table-name {table} \
         --attribute-definitions AttributeName={pk},AttributeType=S AttributeName={sk},AttributeType=S \
         --key-schema AttributeName={pk},KeyType=HASH AttributeName={sk},KeyType=RANGE \
         --billing-mode PAY_PER_REQUEST{cli_options}",
        pk = keys::PK,
        sk = keys::SK,
    )
}

/// `aws dynamodb update-time-to-live` for `table`, then `cli_options`.
pub fn update_time_to_live_command(table: &str, cli_options: &str) -> String {
    format!(
        "aws dynamodb update-time-to-live --table-name {table} \
         --time-to-live-specification Enabled=true,AttributeName={}{cli_options}",
        keys::EXPIRES_AT
    )
}

/// The table, or `None` if it does not exist.
async fn read(client: &Client, table: &str) -> anyhow::Result<Option<TableDescription>> {
    match client.describe_table().table_name(table).send().await {
        Ok(output) => Ok(output.table),
        Err(error)
            if error
                .as_service_error()
                .is_some_and(|error| error.is_resource_not_found_exception()) =>
        {
            Ok(None)
        }
        Err(error) => Err(error).with_context(|| format!("describe DynamoDB table {table}")),
    }
}

/// Waits until `table` is usable: active, or being updated (for example a
/// new Global Table replica), which still serves reads and writes.
async fn wait_until_active(client: &Client, table: &str) -> anyhow::Result<TableDescription> {
    let deadline = tokio::time::Instant::now() + ACTIVE_WITHIN;
    let mut wait = FIRST_CHECK_AFTER;
    loop {
        let description = read(client, table)
            .await?
            .with_context(|| format!("DynamoDB table {table} went away"))?;
        match description.table_status() {
            Some(TableStatus::Active | TableStatus::Updating) => return Ok(description),
            Some(TableStatus::Creating) | None => {}
            Some(status) => bail!("DynamoDB table {table} is {status}, so it cannot be used"),
        }
        if tokio::time::Instant::now() + wait > deadline {
            bail!(
                "DynamoDB table {table} did not become active within {}s",
                ACTIVE_WITHIN.as_secs()
            );
        }
        tokio::time::sleep(wait).await;
        wait = (wait * 2).min(MAX_CHECK_EVERY);
    }
}

/// Fails unless the table's key is `pk` (HASH) and `sk` (RANGE), both
/// strings.
fn check_keys(table: &str, description: &TableDescription) -> anyhow::Result<()> {
    let has_key = |name: &str, key_type: KeyType| {
        description
            .key_schema()
            .iter()
            .any(|key| key.attribute_name() == name && *key.key_type() == key_type)
            && description
                .attribute_definitions()
                .iter()
                .any(|definition| {
                    definition.attribute_name() == name
                        && *definition.attribute_type() == ScalarAttributeType::S
                })
    };
    if description.key_schema().len() != 2
        || !has_key(keys::PK, KeyType::Hash)
        || !has_key(keys::SK, KeyType::Range)
    {
        let found: Vec<String> = description
            .key_schema()
            .iter()
            .map(|key| format!("{} ({})", key.attribute_name(), key.key_type()))
            .collect();
        bail!(
            "DynamoDB table {table} has the key {}, but registrations need {} (HASH) and {} \
             (RANGE), both strings",
            found.join(", "),
            keys::PK,
            keys::SK,
        );
    }
    Ok(())
}

fn attribute(name: &str) -> anyhow::Result<AttributeDefinition> {
    Ok(AttributeDefinition::builder()
        .attribute_name(name)
        .attribute_type(ScalarAttributeType::S)
        .build()?)
}

fn key(name: &str, key_type: KeyType) -> anyhow::Result<KeySchemaElement> {
    Ok(KeySchemaElement::builder()
        .attribute_name(name)
        .key_type(key_type)
        .build()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn description(keys: &[(&str, KeyType, ScalarAttributeType)]) -> TableDescription {
        let mut builder = TableDescription::builder().table_status(TableStatus::Active);
        for (name, key_type, attribute_type) in keys {
            builder = builder
                .key_schema(key(name, key_type.clone()).unwrap())
                .attribute_definitions(
                    AttributeDefinition::builder()
                        .attribute_name(*name)
                        .attribute_type(attribute_type.clone())
                        .build()
                        .unwrap(),
                );
        }
        builder.build()
    }

    #[test]
    fn only_the_designed_key_is_accepted() {
        use ScalarAttributeType::{N, S};
        let designed = description(&[("pk", KeyType::Hash, S), ("sk", KeyType::Range, S)]);
        assert!(check_keys("pools", &designed).is_ok());
        for wrong in [
            description(&[("pk", KeyType::Hash, S)]),
            description(&[("sk", KeyType::Hash, S), ("pk", KeyType::Range, S)]),
            description(&[("pk", KeyType::Hash, S), ("sk", KeyType::Range, N)]),
            description(&[("id", KeyType::Hash, S), ("sk", KeyType::Range, S)]),
        ] {
            let error = check_keys("pools", &wrong).unwrap_err().to_string();
            assert!(
                error.contains("DynamoDB table pools has the key"),
                "{error}"
            );
        }
    }

    #[test]
    fn the_commands_name_the_table() {
        let create = create_table_command("pools", "");
        assert!(create.starts_with("aws dynamodb create-table --table-name pools "));
        assert!(create.contains("AttributeName=pk,KeyType=HASH AttributeName=sk,KeyType=RANGE"));
        assert!(update_time_to_live_command("pools", "").ends_with(
            "--table-name pools --time-to-live-specification Enabled=true,AttributeName=expires_at"
        ));
    }

    /// The commands create the table where the service looks for it, not
    /// where the CLI's own defaults point.
    #[test]
    fn the_commands_name_the_region_and_the_endpoint() {
        let region = Region::new("us-east-2");
        let local = Url::parse("http://localhost:8000").unwrap();
        assert_eq!(cli_options(None, None), "");
        assert_eq!(cli_options(Some(&region), None), " --region us-east-2");
        let options = cli_options(Some(&region), Some(&local));
        assert_eq!(
            options,
            " --region us-east-2 --endpoint-url http://localhost:8000"
        );
        assert!(create_table_command("pools", &options).ends_with(
            "--billing-mode PAY_PER_REQUEST --region us-east-2 --endpoint-url http://localhost:8000"
        ));
        assert!(update_time_to_live_command("pools", &options).ends_with(
            "AttributeName=expires_at --region us-east-2 --endpoint-url http://localhost:8000"
        ));
    }
}
