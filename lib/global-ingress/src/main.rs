// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Global Frontend. See `dynamo_global_ingress`.
//!
//! ```bash
//! cargo run -p dynamo-global-ingress -- --port 8000 \
//!     --pool ohio/dynamo/mocker-1=Qwen/Qwen3-0.6B@http://10.42.0.5:8000
//! ```

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::Parser;
use dynamo_global_ingress::{ModelSync, PoolForwardingEngine};
use dynamo_global_router::{FirstPoolSelector, Pool, PoolCatalog, StaticPoolCatalog};
use dynamo_global_router_shared::{PoolIdDeriver, PoolKey, PoolLocation, V1PoolIdDeriver};
use dynamo_llm::http::service::service_v2::HttpService;
use tokio_util::sync::CancellationToken;
use url::Url;

#[derive(Parser)]
#[command(about = "Serve the OpenAI API and forward each request to a pool frontend")]
struct Args {
    /// Address to serve HTTP on.
    #[arg(long, default_value = "0.0.0.0")]
    host: String,
    #[arg(long, default_value_t = 8000)]
    port: u16,
    /// A pool as `<site>/<namespace>/<DGD name>=<model>@<frontend URL>`.
    /// Separate several frontend URLs with commas, and repeat the flag for
    /// each pool. The site is also the pool's region.
    #[arg(long = "pool", value_parser = parse_pool, required = true)]
    pools: Vec<Pool>,
    /// How long to wait for a pool frontend to accept a connection.
    #[arg(long, default_value_t = 5)]
    connect_timeout_secs: u64,
}

fn parse_pool(value: &str) -> Result<Pool, String> {
    let shape =
        || format!("expected <site>/<namespace>/<DGD name>=<model>@<URL>[,<URL>...], got {value}");
    let (key, served) = value.split_once('=').ok_or_else(shape)?;
    let [site, namespace, dgd] = key.split('/').collect::<Vec<_>>()[..] else {
        return Err(shape());
    };
    let (model, frontends) = served.split_once('@').ok_or_else(shape)?;
    let key = PoolKey::new(site, namespace, dgd).map_err(|error| error.to_string())?;
    let frontends = frontends
        .split(',')
        .map(|url| {
            url.parse::<Url>()
                .map_err(|error| format!("{url}: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Pool {
        id: V1PoolIdDeriver.derive(&key),
        model: model.to_owned(),
        location: PoolLocation {
            region: site.to_owned(),
            availability_zone: None,
            cluster: None,
            datacenter: None,
        },
        frontends,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    dynamo_runtime::logging::init();
    let args = Args::parse();
    let mut ids = HashSet::new();
    if let Some(repeated) = args.pools.iter().find(|pool| !ids.insert(&pool.id)) {
        bail!("--pool {} is given twice", repeated.id);
    }

    let shutdown = CancellationToken::new();
    let signal = shutdown.clone();
    tokio::spawn(async move {
        if let Err(error) = shutdown_signal().await {
            tracing::warn!(%error, "shutdown signal handler failed");
        }
        signal.cancel();
    });

    let catalog = StaticPoolCatalog::new(args.pools);
    let service = HttpService::builder()
        .host(args.host.clone())
        .port(args.port)
        .cancel_token(Some(shutdown.clone()))
        .build()
        .context("build the HTTP service")?;
    let http = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(args.connect_timeout_secs))
        .build()
        .context("build the pool HTTP client")?;
    let engine = Arc::new(PoolForwardingEngine::new(
        catalog.subscribe(),
        Arc::new(FirstPoolSelector),
        http,
    ));

    // Register the models before serving, so the first request finds them.
    let pools = catalog.subscribe();
    let mut models = ModelSync::new(service.clone(), engine);
    models.apply(&pools.borrow());
    tokio::spawn(models.follow(pools, shutdown.child_token()));

    tracing::info!(host = %args.host, port = args.port, "Global Frontend listening");
    service
        .run(shutdown)
        .await
        .context("serve the Global Frontend")
}

async fn shutdown_signal() -> Result<()> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};

        let mut terminate = signal(SignalKind::terminate()).context("register SIGTERM handler")?;
        tokio::select! {
            result = tokio::signal::ctrl_c() => result.context("wait for Ctrl-C")?,
            _ = terminate.recv() => {}
        }
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c().await.context("wait for Ctrl-C")?;
    Ok(())
}
