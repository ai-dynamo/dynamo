// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Data-group tags and priority for Contact Router.
//!
//! A data group is the samples of one prompt that train together. Requests name
//! their group, its size and their task's environment seconds in headers, and
//! carry the weight version the harness is on. A program scores
//! `wv * group_tag + left * programs_left + env_s * env_seconds`, where the group
//! tag is the version its first program started at; higher scores go first.

use std::collections::{BTreeMap, HashMap, HashSet, hash_map::Entry};
use std::time::{Duration, Instant};

use serde::Deserialize;

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PriorityConfig {
    /// Weight on the group's version tag; negative serves older groups first.
    #[serde(default)]
    pub(crate) wv: f64,
    /// Weight on the group's unfinished programs; negative serves nearly done groups first.
    #[serde(default)]
    pub(crate) left: f64,
    /// Weight on the task's environment seconds; positive serves slow environments first.
    #[serde(default)]
    pub(crate) env_s: f64,
}

/// Headers carrying the data group, its size, its environment seconds and
/// the weight version; `dynamo-llm` forwards them as agent headers.
const GROUP_HEADER: &str = "x-dynamo-group";
const GROUP_SIZE_HEADER: &str = "x-dynamo-group-size";
const ENV_HEADER: &str = "x-dynamo-env-s";
pub(crate) const VERSION_HEADER: &str = "x-dynamo-weight-version";

/// The last value of header `name`, trimmed.
pub(crate) fn header(headers: &BTreeMap<String, Vec<String>>, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|values| values.last())
        .map(|value| value.trim().to_string())
}

/// A data group named by `x-dynamo-group`, or the one session of a program
/// that names none; the two never collide.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum GroupKey {
    Named(String),
    Session(String),
}

/// The group a request belongs to, read from its headers.
#[derive(Clone, Debug, Default)]
pub(crate) struct GroupTag {
    pub(crate) group: Option<GroupKey>,
    pub(crate) size: usize,
    pub(crate) env_s: f64,
}

impl PriorityConfig {
    #[cfg(test)]
    pub(crate) fn new(wv: f64, left: f64, env_s: f64) -> Self {
        Self { wv, left, env_s }
    }
}

impl GroupTag {
    pub(crate) fn from_headers(headers: &BTreeMap<String, Vec<String>>) -> Self {
        Self {
            group: header(headers, GROUP_HEADER).map(GroupKey::Named),
            size: header(headers, GROUP_SIZE_HEADER)
                .and_then(|value| value.parse().ok())
                .unwrap_or(1),
            env_s: header(headers, ENV_HEADER)
                .and_then(|value| value.parse().ok())
                .unwrap_or(0.0),
        }
    }
}

/// A data group's tag and how many of its programs have not finished.
struct Group {
    wv: u64,
    left: usize,
    /// Sessions that started a program; a second start is a retry and takes a slot back.
    members: HashSet<String>,
    /// When a program last ended without finishing; the group is kept for its retry.
    dropped_at: Option<Instant>,
    aged: bool,
}

impl Group {
    fn new(tag: &GroupTag, version: u64, session: &str) -> Self {
        Self {
            wv: version,
            left: tag.size.max(1),
            members: HashSet::from([session.to_owned()]),
            dropped_at: None,
            aged: false,
        }
    }
}

#[derive(Default)]
pub(crate) struct Groups(HashMap<GroupKey, Group>);

impl Groups {
    /// Program `session` of `tag` started at `version`; its group keeps the first start's version.
    pub(crate) fn start(&mut self, tag: &GroupTag, version: u64, session: &str) {
        let Some(name) = &tag.group else {
            return;
        };
        match self.0.entry(name.clone()) {
            Entry::Occupied(mut entry) => {
                let group = entry.get_mut();
                if group.left == 0 && !group.members.contains(session) {
                    *group = Group::new(tag, version, session);
                } else if !group.members.insert(session.to_owned()) {
                    group.left += 1;
                }
            }
            Entry::Vacant(entry) => {
                entry.insert(Group::new(tag, version, session));
            }
        }
    }

    /// The version `tag`'s group started at, while any of its programs runs.
    pub(crate) fn wv(&self, tag: &GroupTag) -> Option<u64> {
        self.0
            .get(tag.group.as_ref()?)
            .filter(|group| group.left > 0)
            .map(|group| group.wv)
    }

    /// Whether program `session` of `tag` joins a group that already started:
    /// one still running, or one this session already started.
    pub(crate) fn started(&self, tag: &GroupTag, session: &str) -> bool {
        tag.group
            .as_ref()
            .and_then(|name| self.0.get(name))
            .is_some_and(|group| group.left > 0 || group.members.contains(session))
    }

    /// Programs of started groups that have not ended yet.
    pub(crate) fn programs(&self) -> usize {
        self.0.values().map(|group| group.left).sum()
    }

    /// A program of `tag` ended; its group has one fewer to wait for.
    /// Returns the group's start version once its last program ends.
    pub(crate) fn end(&mut self, tag: &GroupTag) -> Option<u64> {
        let name = tag.group.as_ref()?;
        let group = self.0.get_mut(name)?;
        group.left = group.left.saturating_sub(1);
        if group.left > 0 || group.aged {
            return None;
        }
        group.aged = true;
        let wv = group.wv;
        if group.dropped_at.is_none() {
            self.0.remove(name);
        }
        Some(wv)
    }

    /// A program of `tag` ended without finishing at `now`. Its slot is freed
    /// until its session starts again.
    pub(crate) fn drop_program(&mut self, tag: &GroupTag, now: Instant) {
        let Some(group) = tag.group.as_ref().and_then(|name| self.0.get_mut(name)) else {
            return;
        };
        group.left = group.left.saturating_sub(1);
        group.dropped_at = Some(now);
    }

    /// Forget groups with nothing left to wait for, once no retry is expected:
    /// `retention` after their last dropped program.
    pub(crate) fn expire(&mut self, now: Instant, retention: Duration) {
        self.0.retain(|_, group| {
            group.left > 0
                || group
                    .dropped_at
                    .is_some_and(|at| now.saturating_duration_since(at) < retention)
        });
    }

    /// The score of a program of `tag` tagged `wv`, scored by its group once that started.
    pub(crate) fn score(&self, priority: &PriorityConfig, tag: &GroupTag, wv: u64) -> f64 {
        let (wv, left) = tag
            .group
            .as_ref()
            .and_then(|name| self.0.get(name))
            .map_or((wv, 1), |group| (group.wv, group.left));
        priority.wv * wv as f64 + priority.left * left as f64 + priority.env_s * tag.env_s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_group_keeps_its_first_version_and_counts_down() {
        let priority = PriorityConfig::new(-1.0, -10.0, 0.5);
        let tag = GroupTag {
            group: Some(GroupKey::Named("g".into())),
            size: 2,
            env_s: 4.0,
        };
        let mut groups = Groups::default();
        groups.start(&tag, 3, "a");
        groups.start(&tag, 7, "b");
        assert_eq!(groups.score(&priority, &tag, 7), -3.0 - 20.0 + 2.0);
        groups.end(&tag);
        assert_eq!(groups.score(&priority, &tag, 7), -3.0 - 10.0 + 2.0);
        groups.end(&tag);
        assert_eq!(groups.score(&priority, &tag, 9), -9.0 - 10.0 + 2.0);
    }

    #[test]
    fn a_retried_program_takes_its_slot_back() {
        let tag = GroupTag {
            group: Some(GroupKey::Named("g".into())),
            size: 2,
            env_s: 0.0,
        };
        let now = Instant::now();
        let retention = Duration::from_secs(60);
        let mut groups = Groups::default();
        groups.start(&tag, 3, "a");
        groups.drop_program(&tag, now);
        assert_eq!(groups.programs(), 1);
        assert_eq!(groups.end(&tag), Some(3));
        assert_eq!(groups.programs(), 0);
        assert!(groups.started(&tag, "a") && !groups.started(&tag, "c"));
        groups.start(&tag, 9, "a");
        assert_eq!(groups.wv(&tag), Some(3));
        assert_eq!(groups.end(&tag), None);
        groups.expire(now + retention, retention);
        assert!(groups.0.is_empty());
        groups.start(&tag, 9, "c");
        assert_eq!((groups.wv(&tag), groups.programs()), (Some(9), 2));
        // A retry long after its drop, while another member runs, takes its slot back.
        groups.start(&tag, 9, "d");
        groups.drop_program(&tag, now + retention);
        groups.expire(now + 3 * retention, retention);
        groups.start(&tag, 9, "d");
        assert_eq!(groups.programs(), 2);
    }
}
