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

/// The group a request belongs to, read from its headers.
#[derive(Clone, Debug, Default)]
pub(crate) struct GroupTag {
    pub(crate) group: Option<String>,
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
            group: header(headers, GROUP_HEADER),
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
    /// Programs that ended without finishing; a retry takes its slot back.
    dropped: HashSet<String>,
    aged: bool,
}

#[derive(Default)]
pub(crate) struct Groups(HashMap<String, Group>);

impl Groups {
    /// Program `session` of `tag` started at `version`; its group keeps the first start's version.
    pub(crate) fn start(&mut self, tag: &GroupTag, version: u64, session: &str) {
        let Some(name) = &tag.group else {
            return;
        };
        match self.0.entry(name.clone()) {
            Entry::Occupied(mut entry) => {
                let group = entry.get_mut();
                if group.dropped.remove(session) {
                    group.left += 1;
                }
            }
            Entry::Vacant(entry) => {
                entry.insert(Group {
                    wv: version,
                    left: tag.size.max(1),
                    dropped: HashSet::new(),
                    aged: false,
                });
            }
        }
    }

    /// The version `tag`'s group started at, once any of its programs has.
    pub(crate) fn wv(&self, tag: &GroupTag) -> Option<u64> {
        self.0.get(tag.group.as_ref()?).map(|group| group.wv)
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
        if group.dropped.is_empty() {
            self.0.remove(name);
        }
        Some(wv)
    }

    /// Program `session` of `tag` ended without finishing. Its slot is freed
    /// until the same session starts again; the group is kept while it may.
    pub(crate) fn drop_program(&mut self, tag: &GroupTag, session: &str) {
        let Some(group) = tag.group.as_ref().and_then(|name| self.0.get_mut(name)) else {
            return;
        };
        if group.dropped.insert(session.to_owned()) {
            group.left = group.left.saturating_sub(1);
        }
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
            group: Some("g".into()),
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
            group: Some("g".into()),
            size: 2,
            env_s: 0.0,
        };
        let mut groups = Groups::default();
        groups.start(&tag, 3, "a");
        groups.drop_program(&tag, "a");
        assert_eq!(groups.programs(), 1);
        assert_eq!(groups.end(&tag), Some(3));
        assert_eq!(groups.programs(), 0);
        groups.start(&tag, 9, "a");
        assert_eq!(groups.programs(), 1);
        assert_eq!(groups.end(&tag), None);
        assert_eq!(groups.programs(), 0);
    }
}
