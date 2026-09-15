// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mooncake master leader discovery for shared-cache KV event routing.

mod config;
mod etcd;
mod kubernetes;
mod redis;

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use async_trait::async_trait;
use tokio::sync::Mutex;
use url::Url;

pub(super) use config::MooncakeHaConfig;

const MOONCAKE_LEADER_CACHE_TTL: Duration = Duration::from_secs(1);

#[derive(Debug, Clone, PartialEq, Eq)]
enum MooncakeMasterEntry {
    Direct,
    Etcd {
        endpoints: Vec<String>,
        cluster_id: String,
    },
    Kubernetes {
        namespace: String,
        lease_name: String,
    },
    Redis {
        host: String,
        port: u16,
        cluster_id: String,
        db_index: u8,
    },
    UnsupportedHa {
        scheme: String,
    },
}

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
struct MooncakeLeaderUnavailable {
    message: String,
}

#[async_trait]
pub(super) trait MooncakeLeaderResolver: Send + Sync {
    async fn current_leader(&self) -> anyhow::Result<String>;
}

struct CachedLeader {
    address: String,
    expires_at: Instant,
}

struct ResolverState {
    entry: MooncakeMasterEntry,
    resolver: Arc<dyn MooncakeLeaderResolver>,
    cached_leader: Option<CachedLeader>,
}

#[derive(Clone, Default)]
pub(super) struct MooncakeHaResolver {
    resolver_state: Arc<Mutex<Option<ResolverState>>>,
    fallback_warned: Arc<AtomicBool>,
}

impl MooncakeHaResolver {
    pub(super) fn config_changed(&self) {
        self.fallback_warned.store(false, Ordering::Release);
    }

    pub(super) async fn resolve_event_endpoint(
        &self,
        config: &MooncakeHaConfig,
        configured_endpoint: &str,
        force_refresh: bool,
    ) -> anyhow::Result<String> {
        let Some(master_server_address) = config.master_server_address.as_deref() else {
            return Ok(configured_endpoint.to_string());
        };
        let entry = match parse_mooncake_master_entry(
            master_server_address,
            config.cluster_id.as_deref().unwrap_or("mooncake"),
            config.redis_db_index,
        ) {
            Ok(entry) => entry,
            Err(error) => {
                if !self.fallback_warned.swap(true, Ordering::AcqRel) {
                    tracing::warn!(
                        %master_server_address,
                        %configured_endpoint,
                        %error,
                        "Failed to parse Mooncake HA locator; using configured KV event endpoint"
                    );
                }
                return Ok(configured_endpoint.to_string());
            }
        };
        match &entry {
            MooncakeMasterEntry::Direct => return Ok(configured_endpoint.to_string()),
            MooncakeMasterEntry::UnsupportedHa { scheme } => {
                if !self.fallback_warned.swap(true, Ordering::AcqRel) {
                    tracing::warn!(
                        %scheme,
                        %configured_endpoint,
                        "Mooncake HA backend has no Dynamo resolver; using configured KV event endpoint"
                    );
                }
                return Ok(configured_endpoint.to_string());
            }
            MooncakeMasterEntry::Etcd { .. }
            | MooncakeMasterEntry::Kubernetes { .. }
            | MooncakeMasterEntry::Redis { .. } => {}
        }

        let leader_address = self.current_leader(&entry, force_refresh).await?;
        match mooncake_event_endpoint_for_leader(configured_endpoint, &leader_address) {
            Ok(endpoint) => {
                self.fallback_warned.store(false, Ordering::Release);
                Ok(endpoint)
            }
            Err(error) => {
                if !self.fallback_warned.swap(true, Ordering::AcqRel) {
                    tracing::warn!(
                        %leader_address,
                        %configured_endpoint,
                        %error,
                        "Failed to rewrite Mooncake KV event endpoint for HA leader; using configured endpoint"
                    );
                }
                Ok(configured_endpoint.to_string())
            }
        }
    }

    async fn current_leader(
        &self,
        entry: &MooncakeMasterEntry,
        force_refresh: bool,
    ) -> anyhow::Result<String> {
        let mut resolver_state = self.resolver_state.lock().await;
        let needs_rebuild = resolver_state
            .as_ref()
            .map(|state| state.entry != *entry)
            .unwrap_or(true);
        if needs_rebuild {
            *resolver_state = Some(ResolverState {
                entry: entry.clone(),
                resolver: build_leader_resolver(entry).await?,
                cached_leader: None,
            });
        }

        let state = resolver_state
            .as_mut()
            .expect("resolver state was initialized above");
        let now = Instant::now();
        if !force_refresh
            && let Some(cached) = state.cached_leader.as_ref()
            && cached.expires_at > now
        {
            return Ok(cached.address.clone());
        }

        let address = state.resolver.current_leader().await?;
        state.cached_leader = Some(CachedLeader {
            address: address.clone(),
            expires_at: now + MOONCAKE_LEADER_CACHE_TTL,
        });
        Ok(address)
    }

    #[cfg(test)]
    pub(super) async fn replace_resolver_for_test(
        &self,
        config: &MooncakeHaConfig,
        resolver: Arc<dyn MooncakeLeaderResolver>,
    ) -> anyhow::Result<()> {
        let master_server_address = config
            .master_server_address
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("test HA config has no master locator"))?;
        let entry = parse_mooncake_master_entry(
            master_server_address,
            config.cluster_id.as_deref().unwrap_or("mooncake"),
            config.redis_db_index,
        )?;
        self.resolver_state.lock().await.replace(ResolverState {
            entry,
            resolver,
            cached_leader: None,
        });
        Ok(())
    }
}

fn parse_mooncake_master_entry(
    master_server_address: &str,
    cluster_id: &str,
    redis_db_index: Option<u16>,
) -> anyhow::Result<MooncakeMasterEntry> {
    let address = master_server_address.trim();
    anyhow::ensure!(!address.is_empty(), "Mooncake master address is empty");

    let Some((scheme, connstring)) = address.split_once("://") else {
        return Ok(MooncakeMasterEntry::Direct);
    };

    match scheme {
        "etcd" => {
            let endpoints = connstring
                .split([';', ','])
                .map(str::trim)
                .filter(|endpoint| !endpoint.is_empty())
                .map(|endpoint| {
                    if endpoint.contains("://") {
                        endpoint.to_string()
                    } else {
                        format!("http://{endpoint}")
                    }
                })
                .collect::<Vec<_>>();
            anyhow::ensure!(
                !endpoints.is_empty(),
                "Mooncake etcd locator has no endpoints"
            );

            anyhow::ensure!(
                !cluster_id.is_empty(),
                "Mooncake etcd locator has an empty cluster ID"
            );
            Ok(MooncakeMasterEntry::Etcd {
                endpoints,
                cluster_id: cluster_id.to_string(),
            })
        }
        "k8s" => {
            let (namespace, lease_name) = connstring
                .split_once('/')
                .unwrap_or(("default", connstring));
            anyhow::ensure!(
                !namespace.is_empty() && !lease_name.is_empty() && !lease_name.contains('/'),
                "Mooncake Kubernetes locator must be k8s://[<namespace>/]<lease>"
            );
            Ok(MooncakeMasterEntry::Kubernetes {
                namespace: namespace.to_string(),
                lease_name: lease_name.to_string(),
            })
        }
        "redis" => {
            let (host, port) = redis::parse_mooncake_redis_endpoint(address)?;
            let cluster_id = redis::sanitize_redis_hash_tag(cluster_id);
            anyhow::ensure!(
                !cluster_id.is_empty(),
                "Mooncake Redis locator has an empty cluster ID"
            );
            let db_index = u8::try_from(redis_db_index.unwrap_or(0)).map_err(|_| {
                anyhow::anyhow!("Mooncake Redis DB index must be between 0 and 255")
            })?;
            Ok(MooncakeMasterEntry::Redis {
                host,
                port,
                cluster_id,
                db_index,
            })
        }
        _ => Ok(MooncakeMasterEntry::UnsupportedHa {
            scheme: scheme.to_string(),
        }),
    }
}

async fn build_leader_resolver(
    entry: &MooncakeMasterEntry,
) -> anyhow::Result<Arc<dyn MooncakeLeaderResolver>> {
    match entry {
        MooncakeMasterEntry::Direct => {
            anyhow::bail!("direct Mooncake master address does not need an HA resolver")
        }
        MooncakeMasterEntry::Etcd {
            endpoints,
            cluster_id,
        } => etcd::build_leader_resolver(endpoints.clone(), cluster_id).await,
        MooncakeMasterEntry::Kubernetes {
            namespace,
            lease_name,
        } => kubernetes::build_leader_resolver(namespace, lease_name).await,
        MooncakeMasterEntry::Redis {
            host,
            port,
            cluster_id,
            db_index,
        } => Ok(redis::build_redis_leader_resolver(
            host, *port, cluster_id, *db_index,
        )),
        MooncakeMasterEntry::UnsupportedHa { scheme } => {
            anyhow::bail!("Mooncake HA backend {scheme} has no Dynamo resolver")
        }
    }
}

pub(super) fn is_mooncake_leader_unavailable(error: &anyhow::Error) -> bool {
    error.downcast_ref::<MooncakeLeaderUnavailable>().is_some()
}

#[cfg(test)]
pub(super) fn leader_unavailable_for_test() -> anyhow::Error {
    MooncakeLeaderUnavailable {
        message: "no active leader in test".to_string(),
    }
    .into()
}

pub(super) fn mooncake_event_endpoint_for_leader(
    configured_endpoint: &str,
    leader_rpc_address: &str,
) -> anyhow::Result<String> {
    let mut event_url = Url::parse(configured_endpoint)?;
    anyhow::ensure!(
        event_url.scheme() == "tcp",
        "Mooncake KV event endpoint must use tcp://"
    );
    let leader_url = Url::parse(&format!("tcp://{leader_rpc_address}"))?;
    let leader_host = leader_url
        .host_str()
        .ok_or_else(|| anyhow::anyhow!("Mooncake leader address has no host"))?;
    event_url
        .set_host(Some(leader_host))
        .map_err(|_| anyhow::anyhow!("failed to set Mooncake KV event leader host"))?;
    Ok(event_url.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use k8s_openapi::api::coordination::v1::{Lease, LeaseSpec};
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::MicroTime;

    #[test]
    fn test_parse_mooncake_master_entries() {
        assert_eq!(
            parse_mooncake_master_entry("mooncake-master:50051", "mooncake", None).unwrap(),
            MooncakeMasterEntry::Direct
        );
        assert_eq!(
            parse_mooncake_master_entry(
                "etcd://etcd-0:2379,etcd-1:2379;etcd-2:2379",
                "mooncake-h200-ib-new",
                None,
            )
            .unwrap(),
            MooncakeMasterEntry::Etcd {
                endpoints: vec![
                    "http://etcd-0:2379".to_string(),
                    "http://etcd-1:2379".to_string(),
                    "http://etcd-2:2379".to_string(),
                ],
                cluster_id: "mooncake-h200-ib-new".to_string(),
            }
        );
        assert_eq!(
            parse_mooncake_master_entry("k8s://dynamo/mooncake-master-lease", "ignored", None,)
                .unwrap(),
            MooncakeMasterEntry::Kubernetes {
                namespace: "dynamo".to_string(),
                lease_name: "mooncake-master-lease".to_string(),
            }
        );
        assert_eq!(
            parse_mooncake_master_entry("k8s://mooncake-master-lease", "ignored", None).unwrap(),
            MooncakeMasterEntry::Kubernetes {
                namespace: "default".to_string(),
                lease_name: "mooncake-master-lease".to_string(),
            }
        );
        assert_eq!(
            parse_mooncake_master_entry("redis://redis:6379", "cluster/{a}", Some(7)).unwrap(),
            MooncakeMasterEntry::Redis {
                host: "redis".to_string(),
                port: 6379,
                cluster_id: "cluster/_a_".to_string(),
                db_index: 7,
            }
        );
    }

    #[test]
    fn test_mooncake_master_view_key_matches_mooncake() {
        assert_eq!(
            etcd::mooncake_master_view_key("mooncake-h200-ib-new"),
            "mooncake-store/mooncake-h200-ib-new/master_view"
        );
        assert_eq!(
            etcd::mooncake_master_view_key("mooncake-h200-ib-new/"),
            "mooncake-store/mooncake-h200-ib-new/master_view"
        );
    }

    #[test]
    fn test_mooncake_leader_from_kubernetes_lease() {
        let now = Utc::now();
        let lease = Lease {
            spec: Some(LeaseSpec {
                holder_identity: Some("10.0.0.4:50051".to_string()),
                renew_time: Some(MicroTime(now - chrono::Duration::seconds(4))),
                lease_duration_seconds: Some(5),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(
            kubernetes::mooncake_leader_from_lease_at(&lease, now).as_deref(),
            Some("10.0.0.4:50051")
        );
    }

    #[test]
    fn test_expired_kubernetes_lease_has_no_leader() {
        let now = Utc::now();
        let lease = Lease {
            spec: Some(LeaseSpec {
                holder_identity: Some("10.0.0.4:50051".to_string()),
                renew_time: Some(MicroTime(now - chrono::Duration::seconds(6))),
                lease_duration_seconds: Some(5),
                ..Default::default()
            }),
            ..Default::default()
        };

        assert_eq!(kubernetes::mooncake_leader_from_lease_at(&lease, now), None);
    }

    #[test]
    fn test_mooncake_event_endpoint_replaces_only_leader_host() {
        assert_eq!(
            mooncake_event_endpoint_for_leader(
                "tcp://mooncake-master.internal:5557",
                "10.0.0.4:50051",
            )
            .unwrap(),
            "tcp://10.0.0.4:5557"
        );
    }

    #[test]
    fn test_mooncake_event_endpoint_accepts_ipv6_leader() {
        assert_eq!(
            mooncake_event_endpoint_for_leader(
                "tcp://mooncake-master.internal:5557",
                "[2001:db8::4]:50051",
            )
            .unwrap(),
            "tcp://[2001:db8::4]:5557"
        );
    }
}
