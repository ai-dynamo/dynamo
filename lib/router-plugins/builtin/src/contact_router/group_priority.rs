// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Data-group tags and priority for Contact Router.
//!
//! A data group is the samples of one prompt that train together. Requests name
//! their group, its size and their task's environment seconds in headers, and
//! carry the weight version the harness is on. A program scores
//! `wv * group_tag + left * programs_left + env_s * env_seconds`, where the group
//! tag is the version its first program started at; higher scores go first.

use std::collections::{BTreeMap, HashMap};

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
    /// Headers carrying the data group, its size and its environment seconds.
    #[serde(default = "default_group_header")]
    group_header: String,
    #[serde(default = "default_group_size_header")]
    group_size_header: String,
    #[serde(default = "default_env_header")]
    env_header: String,
}

fn default_group_header() -> String {
    "x-dynamo-group".into()
}

fn default_group_size_header() -> String {
    "x-dynamo-group-size".into()
}

fn default_env_header() -> String {
    "x-dynamo-env-s".into()
}

pub(crate) fn default_version_header() -> String {
    "x-dynamo-weight-version".into()
}

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
    pub(crate) fn new(wv: f64, left: f64, env_s: f64) -> Self {
        Self {
            wv,
            left,
            env_s,
            group_header: default_group_header(),
            group_size_header: default_group_size_header(),
            env_header: default_env_header(),
        }
    }

    pub(crate) fn tag(&self, headers: &BTreeMap<String, Vec<String>>) -> GroupTag {
        GroupTag {
            group: header(headers, &self.group_header),
            size: header(headers, &self.group_size_header)
                .and_then(|value| value.parse().ok())
                .unwrap_or(1),
            env_s: header(headers, &self.env_header)
                .and_then(|value| value.parse().ok())
                .unwrap_or(0.0),
        }
    }
}

/// A data group's tag and how many of its programs have not finished.
struct Group {
    wv: u64,
    left: usize,
}

#[derive(Default)]
pub(crate) struct Groups(HashMap<String, Group>);

impl Groups {
    /// A program of `tag` started at `version`; its group keeps the first start's version.
    pub(crate) fn start(&mut self, tag: &GroupTag, version: u64) {
        if let Some(group) = &tag.group {
            self.0.entry(group.clone()).or_insert(Group {
                wv: version,
                left: tag.size.max(1),
            });
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
        (group.left == 0).then(|| self.0.remove(name).map(|group| group.wv))?
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
        groups.start(&tag, 3);
        groups.start(&tag, 7);
        assert_eq!(groups.score(&priority, &tag, 7), -3.0 - 20.0 + 2.0);
        groups.end(&tag);
        assert_eq!(groups.score(&priority, &tag, 7), -3.0 - 10.0 + 2.0);
        groups.end(&tag);
        assert_eq!(groups.score(&priority, &tag, 9), -9.0 - 10.0 + 2.0);
    }
}
