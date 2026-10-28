// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Global Frontend. See `dynamo_global_frontend`.
//!
//! ```bash
//! cargo run -p dynamo-global-frontend -- --port 8000 \
//!     --pool ohio/dynamo/mocker-1=Qwen/Qwen3-0.6B@http://10.42.0.5:8000
//! ```
//!
//! With session affinity, each pool's load comes from its KV DC Relay:
//!
//! ```bash
//! cargo run -p dynamo-global-frontend -- --port 8000 \
//!     --router-policy affinity --region ohio \
//!     --pool ohio/dynamo/mocker-1=Qwen/Qwen3-0.6B@http://10.42.0.5:8000 \
//!     --pool-relay ohio/dynamo/mocker-1=http://10.42.0.6:50051
//! ```

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use dynamo_global_common::{PoolIdDeriver, PoolKey, PoolLocation, V1PoolIdDeriver};
use dynamo_global_frontend::{ModelSync, PoolForwardingEngine, follow_relays, measure_round_trips};
use dynamo_global_frontend::{
    Pool, PoolCatalog, PoolSelector, RoundRobinSelector, StaticPoolCatalog,
};
use dynamo_global_router::{AffinitySelector, InFlightTracker, LeastInFlightSelector, PoolSignals};
use dynamo_llm::http::service::service_v2::HttpService;
use tokio_util::sync::CancellationToken;
use url::Url;

/// A load report older than this is not used: three of the Relay's reports,
/// which come about once a second.
const LOAD_MAX_AGE: Duration = Duration::from_secs(3);
/// How often the round trip to each pool is measured.
const ROUND_TRIP_INTERVAL: Duration = Duration::from_secs(30);

/// How each request picks a pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum RouterPolicy {
    /// Every pool serving the model in turn.
    RoundRobin,
    /// The pool with the fewest of this process's requests in flight.
    LeastInFlight,
    /// Keep each session on one pool, nearest region first, and move it only
    /// for a pool with much more room (see `--headroom-factor`). Reads each
    /// pool's load from its Relay (`--pool-relay`).
    Affinity,
}

#[derive(Parser)]
#[command(about = "Serve the OpenAI API and forward each request to a pool frontend")]
struct Args {
    /// Address to serve HTTP on.
    #[arg(long, default_value = "0.0.0.0")]
    host: String,
    #[arg(long, default_value_t = 8000)]
    port: u16,
    /// A pool as `<site>/<namespace>/<DGD name>=<model>@<frontend URL>`.
    /// Separate several frontend URLs with commas,
    /// and repeat the flag for each pool. The site is also the pool's region.
    #[arg(long = "pool", value_parser = parse_pool, required = true)]
    pools: Vec<Pool>,
    /// How long to wait for a pool frontend to accept a connection.
    #[arg(long, default_value_t = 5)]
    connect_timeout_secs: u64,
    #[arg(long, value_enum, default_value_t = RouterPolicy::RoundRobin)]
    router_policy: RouterPolicy,
    /// The region this process runs in, named as `--pool` names sites.
    /// `affinity` tries pools in it first.
    #[arg(long, required_if_eq("router_policy", "affinity"))]
    region: Option<String>,
    /// A pool's KV DC Relay, which streams the pool's load, as
    /// `<site>/<namespace>/<DGD name>=<URL>`. Repeat the flag for each pool.
    #[arg(long = "pool-relay", value_parser = parse_pool_relay)]
    pool_relays: Vec<(PoolKey, Url)>,
    /// `affinity`: a request leaves its pool only for one with more than this
    /// many times its spare capacity, or when its pool is full.
    #[arg(long, default_value_t = 2.0)]
    headroom_factor: f64,
    /// `affinity`: how long a session keeps its pool without a request.
    #[arg(long, default_value_t = 1800)]
    session_idle_secs: u64,
}

fn parse_key(key: &str) -> Result<PoolKey, String> {
    let [site, namespace, dgd] = key.split('/').collect::<Vec<_>>()[..] else {
        return Err(format!("expected <site>/<namespace>/<DGD name>, got {key}"));
    };
    PoolKey::new(site, namespace, dgd).map_err(|error| error.to_string())
}

fn parse_url(url: &str, what: &str) -> Result<Url, String> {
    let parsed = url
        .parse::<Url>()
        .map_err(|error| format!("{url}: {error}"))?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        return Err(format!("{what} must be an HTTP(S) URL: {url}"));
    }
    Ok(parsed)
}

fn parse_pool_relay(value: &str) -> Result<(PoolKey, Url), String> {
    let (key, relay) = value
        .split_once('=')
        .ok_or_else(|| format!("expected <site>/<namespace>/<DGD name>=<URL>, got {value}"))?;
    Ok((parse_key(key)?, parse_url(relay, "pool relay")?))
}

fn parse_pool(value: &str) -> Result<Pool, String> {
    let shape =
        || format!("expected <site>/<namespace>/<DGD name>=<model>@<URL>[,<URL>...], got {value}");
    let (key, served) = value.split_once('=').ok_or_else(shape)?;
    let (model, frontends) = served.split_once('@').ok_or_else(shape)?;
    if model.trim().is_empty() {
        return Err("pool model must not be empty".to_owned());
    }
    let key = parse_key(key)?;
    let frontends = frontends
        .split(',')
        .map(|url| parse_url(url, "pool frontend"))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Pool {
        id: V1PoolIdDeriver.derive(&key),
        model: model.to_owned(),
        location: PoolLocation {
            region: key.site_id().to_owned(),
            availability_zone: None,
            cluster: None,
            datacenter: None,
        },
        frontends,
        relay: None,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    dynamo_runtime::logging::init();
    let mut args = Args::parse();
    let mut ids = HashSet::new();
    if let Some(repeated) = args.pools.iter().find(|pool| !ids.insert(&pool.id)) {
        bail!("--pool {} is given twice", repeated.id);
    }
    for (key, relay) in &args.pool_relays {
        let id = V1PoolIdDeriver.derive(key);
        let Some(pool) = args.pools.iter_mut().find(|pool| pool.id == id) else {
            bail!("--pool-relay {relay} names a pool no --pool gives");
        };
        if pool.relay.replace(relay.clone()).is_some() {
            bail!("--pool-relay is given twice for pool {id}");
        }
    }
    if !(args.headroom_factor >= 1.0 && args.headroom_factor.is_finite()) {
        bail!("--headroom-factor must be at least 1");
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
    let in_flight = Arc::new(InFlightTracker::new());
    let selector: Arc<dyn PoolSelector> = match args.router_policy {
        RouterPolicy::RoundRobin => Arc::new(RoundRobinSelector::new()),
        RouterPolicy::LeastInFlight => Arc::new(LeastInFlightSelector::new(in_flight.clone())),
        RouterPolicy::Affinity => {
            let region = args.region.clone().context("--region is required")?;
            if args.pool_relays.is_empty() {
                tracing::warn!(
                    "no --pool-relay is given, so affinity knows no pool's load \
                     and keeps each session on its nearest pool"
                );
            }
            let signals = Arc::new(PoolSignals::new(in_flight.clone(), LOAD_MAX_AGE));
            tokio::spawn(follow_relays(
                catalog.subscribe(),
                signals.clone(),
                shutdown.child_token(),
            ));
            tokio::spawn(measure_round_trips(
                catalog.subscribe(),
                signals.clone(),
                ROUND_TRIP_INTERVAL,
                shutdown.child_token(),
            ));
            let idle = Duration::from_secs(args.session_idle_secs);
            Arc::new(AffinitySelector::new(
                signals,
                region,
                args.headroom_factor,
                idle,
            ))
        }
    };
    let engine = Arc::new(PoolForwardingEngine::new(
        catalog.subscribe(),
        selector,
        in_flight,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_model_paths_and_multiple_frontends() {
        let pool =
            parse_pool("ohio/dynamo/blue=Qwen/Qwen3@http://blue:8000,https://green:8000").unwrap();
        assert_eq!(pool.model, "Qwen/Qwen3");
        assert_eq!(pool.frontends.len(), 2);
    }

    #[test]
    fn rejects_missing_models_and_non_http_frontends() {
        for value in [
            "site/ns/pool=@http://pool:8000",
            "site/ns/pool=m@file:///tmp/pool",
            "site/ns/pool=m@mailto:user@example.com",
        ] {
            assert!(parse_pool(value).is_err(), "{value}");
        }
    }

    #[test]
    fn parses_pool_relays_and_requires_a_region_for_affinity() {
        let (key, relay) = parse_pool_relay("ohio/dynamo/blue=http://relay:50051").unwrap();
        assert_eq!(key.dgd_name(), "blue");
        assert_eq!(relay.as_str(), "http://relay:50051/");
        assert!(parse_pool_relay("ohio/dynamo=http://relay:50051").is_err());

        let pool = "--pool=ohio/dynamo/blue=qwen@http://blue:8000";
        let affinity = ["global-frontend", pool, "--router-policy", "affinity"];
        assert!(Args::try_parse_from(affinity).is_err());
        let args = Args::try_parse_from([&affinity[..], &["--region", "ohio"]].concat()).unwrap();
        assert_eq!(args.router_policy, RouterPolicy::Affinity);
        let args = Args::try_parse_from(["global-frontend", pool]).unwrap();
        assert_eq!(args.router_policy, RouterPolicy::RoundRobin);
    }

    #[test]
    fn requires_explicit_pool_configuration() {
        assert!(Args::try_parse_from(["global-frontend"]).is_err());
        assert!(
            Args::try_parse_from([
                "global-frontend",
                "--discovery-url",
                "http://discovery:8080"
            ])
            .is_err()
        );
    }
}
