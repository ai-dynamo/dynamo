// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request deltas between the Global Frontend and pool frontends.
//!
//! A conversation's next turn sends the whole conversation again. A pool
//! frontend that keeps each conversation's last request lets its sender send
//! only what changed: the elements after those it keeps of the request's list
//! (`messages`, or a Responses `input` array), and the request's other fields
//! only when they changed. The pool rebuilds the request from the two.
//!
//! | Request header | Body | Pool |
//! |---|---|---|
//! | [`STORE_HEADER`] | the request | keeps it |
//! | [`DELTA_HEADER`] ([`Delta`]) | [`Parent::delta`]'s body | rebuilds the request with [`State::apply`], and keeps it; 409 if it has no such parent |
//!
//! The pool answers with [`STATE_HEADER`]: the [`Digest`] of the request it
//! kept, or [`MISSING`] with its 409.
//!
//! Digests are blake3, cut to 128 bits, of the elements' raw JSON, so a
//! parent names exactly the bytes the pool keeps, and of the request's
//! `authorization` header, so a delta extends only a parent sent with the
//! same credentials.

use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use serde::Deserialize;
use serde::de::{Deserializer, MapAccess, Visitor};
use serde_json::value::RawValue;
use thiserror::Error;

/// Asks the pool to keep this request, which carries no delta.
pub const STORE_HEADER: &str = "x-dynamo-delta-store";
/// Marks a delta, and names its parent: see [`Delta`].
pub const DELTA_HEADER: &str = "x-dynamo-delta";
/// The pool's answer: the [`Digest`] of the request it kept, or [`MISSING`].
pub const STATE_HEADER: &str = "x-dynamo-delta-state";
/// [`STATE_HEADER`] on a 409: the pool does not have the delta's parent.
pub const MISSING: &str = "missing";

/// Change the version whenever the digests change.
const DERIVE_KEY: &str = "dynamo.global-router.request-delta.v1";
/// The fields that hold a request's list, in the order they are looked for.
const LISTS: [&str; 2] = ["messages", "input"];
/// How many of a parent's last elements a request may replace and still be
/// sent as a delta: clients change their last messages' cache breakpoints.
const TAIL: usize = 16;

/// A 128-bit digest, written as 32 hex digits.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Digest([u8; 16]);

impl Digest {
    fn of(hasher: &blake3::Hasher) -> Self {
        let mut digest = [0; 16];
        digest.copy_from_slice(&hasher.finalize().as_bytes()[..16]);
        Self(digest)
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.iter().try_for_each(|byte| write!(f, "{byte:02x}"))
    }
}

impl fmt::Debug for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl FromStr for Digest {
    type Err = Malformed;

    fn from_str(hex: &str) -> Result<Self, Malformed> {
        if hex.len() != 32 || !hex.is_ascii() {
            return Err(Malformed);
        }
        let mut digest = [0; 16];
        for (byte, pair) in digest.iter_mut().zip(hex.as_bytes().chunks(2)) {
            let pair = std::str::from_utf8(pair).map_err(|_| Malformed)?;
            *byte = u8::from_str_radix(pair, 16).map_err(|_| Malformed)?;
        }
        Ok(Self(digest))
    }
}

/// A [`Digest`] or [`Delta`] that cannot be read.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[error("malformed request delta")]
pub struct Malformed;

fn hasher() -> blake3::Hasher {
    blake3::Hasher::new_derive_key(DERIVE_KEY)
}

/// The link after `previous` for an element whose raw JSON is `element`.
fn link(previous: Digest, element: &str) -> Digest {
    Digest::of(hasher().update(&previous.0).update(element.as_bytes()))
}

/// The digest of a request kept with `rest` and `last` link, sent with
/// `authorization`.
fn state(rest: Digest, last: Digest, authorization: Option<&[u8]>) -> Digest {
    let mut hasher = hasher();
    hasher.update(b"state").update(&rest.0).update(&last.0);
    Digest::of(hasher.update(authorization.unwrap_or_default()))
}

fn credentials(authorization: Option<&[u8]>) -> Digest {
    Digest::of(
        hasher()
            .update(b"authorization")
            .update(authorization.unwrap_or_default()),
    )
}

/// A request body split into its list and its other fields, borrowed from
/// the body, with the digests that name them.
#[derive(Debug)]
pub struct Request<'a> {
    list: &'static str,
    elements: Vec<&'a RawValue>,
    /// The other top-level fields, in the body's order.
    rest: Vec<(String, &'a RawValue)>,
    rest_digest: Digest,
    /// `links[0]` names the list; `links[i]` the list's first `i` elements.
    links: Vec<Digest>,
}

impl<'a> Request<'a> {
    /// `None` unless `body` is a JSON object with a list: a `messages` or
    /// `input` array.
    pub fn parse(body: &'a [u8]) -> Option<Self> {
        let Fields(fields) = serde_json::from_slice(body).ok()?;
        let mut list = None;
        let mut rest = Vec::with_capacity(fields.len());
        for (name, value) in fields {
            if list.is_none()
                && let Some(field) = LISTS.into_iter().find(|field| *field == name)
                && let Ok(elements) = serde_json::from_str::<Vec<&RawValue>>(value.get())
            {
                list = Some((field, elements));
            } else {
                rest.push((name, value));
            }
        }
        let (list, elements) = list?;
        let mut sorted = rest.iter().collect::<Vec<_>>();
        sorted.sort_by(|left, right| left.0.cmp(&right.0));
        let mut rest_digest = hasher();
        rest_digest.update(b"rest");
        for (name, value) in sorted {
            rest_digest.update(&(name.len() as u64).to_le_bytes());
            rest_digest.update(name.as_bytes());
            rest_digest.update(&(value.get().len() as u64).to_le_bytes());
            rest_digest.update(value.get().as_bytes());
        }
        let mut links = Vec::with_capacity(elements.len() + 1);
        links.push(Digest::of(hasher().update(b"list").update(list.as_bytes())));
        for element in &elements {
            links.push(link(links[links.len() - 1], element.get()));
        }
        Some(Self {
            list,
            elements,
            rest,
            rest_digest: Digest::of(&rest_digest),
            links,
        })
    }

    /// The digest the pool keeps this request under, sent with
    /// `authorization`.
    pub fn state(&self, authorization: Option<&[u8]>) -> Digest {
        state(
            self.rest_digest,
            self.links[self.links.len() - 1],
            authorization,
        )
    }

    /// The other fields as `"name":value,` pairs.
    fn rest_text(&self) -> String {
        let mut text = String::new();
        for (name, value) in &self.rest {
            text.push_str(&serde_json::Value::from(name.as_str()).to_string());
            text.push(':');
            text.push_str(value.get());
            text.push(',');
        }
        text
    }
}

/// A JSON object's fields, in order, with their raw values.
struct Fields<'a>(Vec<(String, &'a RawValue)>);

impl<'de: 'a, 'a> Deserialize<'de> for Fields<'a> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct FieldsVisitor;

        impl<'de> Visitor<'de> for FieldsVisitor {
            type Value = Fields<'de>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a JSON object")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Fields<'de>, A::Error> {
                let mut fields = Vec::new();
                while let Some(field) = map.next_entry::<String, &'de RawValue>()? {
                    fields.push(field);
                }
                Ok(Fields(fields))
            }
        }

        deserializer.deserialize_map(FieldsVisitor)
    }
}

/// Where a delta's fields other than the list come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rest {
    /// The parent's, unchanged.
    Parent,
    /// The delta body's.
    Body,
}

/// The [`DELTA_HEADER`] of a delta: the request is the parent's first
/// `keep` elements, then the delta body's, and its other fields come from
/// `rest`. `check` is the parent's link after `keep` elements.
///
/// Written `parent=<digest>; keep=<count>; check=<digest>; rest=parent|body`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Delta {
    pub parent: Digest,
    pub keep: usize,
    pub check: Digest,
    pub rest: Rest,
}

impl fmt::Display for Delta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rest = match self.rest {
            Rest::Parent => "parent",
            Rest::Body => "body",
        };
        write!(
            f,
            "parent={}; keep={}; check={}; rest={rest}",
            self.parent, self.keep, self.check
        )
    }
}

impl FromStr for Delta {
    type Err = Malformed;

    fn from_str(header: &str) -> Result<Self, Malformed> {
        let (mut parent, mut keep, mut check, mut rest) = (None, None, None, None);
        for part in header.split(';') {
            let (name, value) = part.trim().split_once('=').ok_or(Malformed)?;
            match name {
                "parent" => parent = Some(value.parse()?),
                "keep" => keep = Some(value.parse().map_err(|_| Malformed)?),
                "check" => check = Some(value.parse()?),
                "rest" => {
                    rest = Some(match value {
                        "parent" => Rest::Parent,
                        "body" => Rest::Body,
                        _ => return Err(Malformed),
                    })
                }
                _ => return Err(Malformed),
            }
        }
        Ok(Self {
            parent: parent.ok_or(Malformed)?,
            keep: keep.ok_or(Malformed)?,
            check: check.ok_or(Malformed)?,
            rest: rest.ok_or(Malformed)?,
        })
    }
}

/// What a sender remembers of a request a pool kept: enough to cut the next
/// request's delta against it, without the request.
#[derive(Clone, Debug)]
pub struct Parent {
    /// The digest the pool keeps the request under.
    pub state: Digest,
    list: &'static str,
    rest: Digest,
    /// Index in the request's links of `tail[0]`.
    start: usize,
    /// The request's last links, up to [`TAIL`], ending with its whole list.
    tail: Vec<Digest>,
}

impl Parent {
    /// `request`, kept by a pool under `state`.
    pub fn new(state: Digest, request: &Request) -> Self {
        // Keeping no element saves nothing, so links[0] is never a tail.
        let start = request.links.len().saturating_sub(TAIL).max(1);
        Self {
            state,
            list: request.list,
            rest: request.rest_digest,
            start,
            tail: request.links[start..].to_vec(),
        }
    }

    /// `request` as a delta against this parent: its header and body. `None`
    /// unless `request` begins with one of this parent's last [`TAIL`]
    /// prefixes.
    pub fn delta(&self, request: &Request) -> Option<(Delta, Vec<u8>)> {
        if request.list != self.list {
            return None;
        }
        let keep = (self.start..self.start + self.tail.len())
            .rev()
            .find(|&keep| request.links.get(keep) == Some(&self.tail[keep - self.start]))?;
        let rest = if request.rest_digest == self.rest {
            Rest::Parent
        } else {
            Rest::Body
        };
        let mut body = String::from("{");
        if rest == Rest::Body {
            body.push_str(&request.rest_text());
        }
        let elements = request.elements[keep..].iter().map(|element| element.get());
        push_list(&mut body, request.list, elements);
        let delta = Delta {
            parent: self.state,
            keep,
            check: request.links[keep],
            rest,
        };
        Some((delta, body.into_bytes()))
    }
}

/// Appends `"list":[elements]}`.
fn push_list<'e>(body: &mut String, list: &str, elements: impl Iterator<Item = &'e str>) {
    body.push('"');
    body.push_str(list);
    body.push_str("\":[");
    for (index, element) in elements.enumerate() {
        if index > 0 {
            body.push(',');
        }
        body.push_str(element);
    }
    body.push_str("]}");
}

/// A request a pool keeps, to rebuild the next request of its conversation
/// from a delta.
#[derive(Debug)]
pub struct State {
    /// The digest the request is kept under.
    pub id: Digest,
    credentials: Digest,
    list: &'static str,
    /// The fields other than the list, as `"name":value,` pairs.
    rest: Arc<str>,
    rest_digest: Digest,
    elements: Vec<Arc<str>>,
    links: Vec<Digest>,
}

impl State {
    /// `request`, sent with `authorization`.
    pub fn new(request: &Request, authorization: Option<&[u8]>) -> Self {
        Self {
            id: request.state(authorization),
            credentials: credentials(authorization),
            list: request.list,
            rest: request.rest_text().into(),
            rest_digest: request.rest_digest,
            elements: request
                .elements
                .iter()
                .map(|element| Arc::from(element.get()))
                .collect(),
            links: request.links.clone(),
        }
    }

    /// The request `delta`, with body `body` and sent with `authorization`,
    /// makes of this one. `None` if it does not extend this request: other
    /// credentials, another list, or a `keep` this request's links do not
    /// `check`. The kept elements are shared with this request.
    pub fn apply(
        &self,
        delta: &Delta,
        body: &Request,
        authorization: Option<&[u8]>,
    ) -> Option<Self> {
        if credentials(authorization) != self.credentials
            || body.list != self.list
            || self.links.get(delta.keep) != Some(&delta.check)
        {
            return None;
        }
        let (rest, rest_digest) = match delta.rest {
            Rest::Parent => (self.rest.clone(), self.rest_digest),
            Rest::Body => (body.rest_text().into(), body.rest_digest),
        };
        let mut elements = self.elements[..delta.keep].to_vec();
        let mut links = self.links[..=delta.keep].to_vec();
        for element in &body.elements {
            elements.push(Arc::from(element.get()));
            links.push(link(links[links.len() - 1], element.get()));
        }
        Some(Self {
            id: state(rest_digest, links[links.len() - 1], authorization),
            credentials: self.credentials,
            list: self.list,
            rest,
            rest_digest,
            elements,
            links,
        })
    }

    /// The request, as a JSON body: its other fields, then its list.
    pub fn body(&self) -> Vec<u8> {
        let mut body = String::with_capacity(self.weight());
        body.push('{');
        body.push_str(&self.rest);
        push_list(
            &mut body,
            self.list,
            self.elements.iter().map(|element| &**element),
        );
        body.into_bytes()
    }

    /// Roughly the bytes the request holds.
    pub fn weight(&self) -> usize {
        let elements = self.elements.iter().map(|element| element.len() + 1);
        self.rest.len() + elements.sum::<usize>() + self.links.len() * 16 + 16
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const AUTH: Option<&[u8]> = Some(b"Bearer one");

    fn parse(body: &str) -> Request<'_> {
        Request::parse(body.as_bytes()).expect("a request with a list")
    }

    fn value(body: &[u8]) -> Value {
        serde_json::from_slice(body).expect("JSON")
    }

    /// A pool keeps `first`, then gets `second` as a delta against it: the
    /// delta's header and body, and the request the pool rebuilds.
    fn turn(first: &str, second: &str) -> (Delta, Vec<u8>, State) {
        let first = parse(first);
        let kept = State::new(&first, AUTH);
        let parent = Parent::new(first.state(AUTH), &first);
        assert_eq!(parent.state, kept.id);
        let second_request = parse(second);
        let (delta, body) = parent.delta(&second_request).expect("a delta");
        let header = delta.to_string();
        assert_eq!(header.parse::<Delta>(), Ok(delta));
        let rebuilt = kept
            .apply(&delta, &parse(std::str::from_utf8(&body).unwrap()), AUTH)
            .expect("the delta extends the kept request");
        assert_eq!(value(&rebuilt.body()), value(second.as_bytes()));
        assert_eq!(rebuilt.id, second_request.state(AUTH));
        (delta, body, rebuilt)
    }

    #[test]
    fn a_delta_rebuilds_the_next_turn() {
        let turns = [
            (
                r#"{"model":"m","messages":[{"role":"system","content":"s"},{"role":"user","content":"hi"}],"stream":true}"#,
                r#"{"model":"m","messages":[{"role":"system","content":"s"},{"role":"user","content":"hi"},{"role":"assistant","content":"yo"},{"role":"user","content":"more"}],"stream":true}"#,
            ),
            (
                r#"{"model":"m","instructions":"s","input":[{"role":"user","content":"hi"}]}"#,
                r#"{"model":"m","instructions":"s","input":[{"role":"user","content":"hi"},{"type":"reasoning","summary":[]},{"type":"message","role":"assistant","content":"yo"},{"role":"user","content":"more"}]}"#,
            ),
            (
                r#"{"model":"m","system":[{"type":"text","text":"s"}],"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"max_tokens":9}"#,
                r#"{"model":"m","system":[{"type":"text","text":"s"}],"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]},{"role":"assistant","content":"yo"},{"role":"user","content":[{"type":"text","text":"more","cache_control":{"type":"ephemeral"}}]}],"max_tokens":9}"#,
            ),
        ];
        let kept = [2, 1, 1];
        for ((first, second), kept) in turns.into_iter().zip(kept) {
            let (delta, body, _) = turn(first, second);
            assert_eq!(delta.keep, kept, "{second}");
            assert_eq!(delta.rest, Rest::Parent);
            assert!(!value(&body).as_object().unwrap().contains_key("model"));
        }
    }

    #[test]
    fn a_moved_cache_breakpoint_resends_only_its_message() {
        let first = r#"{"model":"m","messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"},{"role":"user","content":[{"type":"text","text":"c","cache_control":{"type":"ephemeral"}}]}]}"#;
        let second = r#"{"model":"m","messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"},{"role":"user","content":[{"type":"text","text":"c"}]},{"role":"assistant","content":"d"}]}"#;
        let (delta, body, _) = turn(first, second);
        assert_eq!(delta.keep, 2);
        assert_eq!(value(&body)["messages"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn changed_fields_ride_along() {
        let first = r#"{"model":"m","tools":[1],"messages":[{"role":"user","content":"a"}]}"#;
        let second = r#"{"model":"m","tools":[2],"messages":[{"role":"user","content":"a"},{"role":"user","content":"b"}]}"#;
        let (delta, body, rebuilt) = turn(first, second);
        assert_eq!(delta.rest, Rest::Body);
        assert_eq!(value(&body)["tools"], serde_json::json!([2]));

        // The rebuilt request keeps the new fields for the turn after.
        let third = r#"{"model":"m","tools":[2],"messages":[{"role":"user","content":"a"},{"role":"user","content":"b"},{"role":"user","content":"c"}]}"#;
        let second = parse(second);
        let (delta, body) = Parent::new(rebuilt.id, &second)
            .delta(&parse(third))
            .unwrap();
        assert_eq!(delta.rest, Rest::Parent);
        let body = std::str::from_utf8(&body).unwrap();
        let next = rebuilt.apply(&delta, &parse(body), AUTH).unwrap();
        assert_eq!(value(&next.body()), value(third.as_bytes()));
    }

    #[test]
    fn a_delta_extends_only_its_own_parent() {
        let first = r#"{"model":"m","messages":[{"role":"user","content":"a"}]}"#;
        let second = r#"{"model":"m","messages":[{"role":"user","content":"a"},{"role":"user","content":"b"}]}"#;
        let first = parse(first);
        let kept = State::new(&first, AUTH);
        let (delta, body) = Parent::new(kept.id, &first).delta(&parse(second)).unwrap();
        let body = std::str::from_utf8(&body).unwrap();
        let body = parse(body);

        assert!(kept.apply(&delta, &body, Some(b"Bearer two")).is_none());
        assert!(kept.apply(&delta, &body, None).is_none());
        let wrong_check = Delta {
            check: kept.links[0],
            ..delta
        };
        assert!(kept.apply(&wrong_check, &body, AUTH).is_none());
        let past_the_end = Delta { keep: 2, ..delta };
        assert!(kept.apply(&past_the_end, &body, AUTH).is_none());
        let other_list = parse(r#"{"input":[{"role":"user","content":"b"}]}"#);
        assert!(kept.apply(&delta, &other_list, AUTH).is_none());
        // Other credentials name another state.
        assert_ne!(first.state(Some(b"Bearer two")), kept.id);
    }

    #[test]
    fn requests_that_diverge_early_are_sent_whole() {
        let first = parse(r#"{"model":"m","messages":[{"role":"user","content":"a"}]}"#);
        let parent = Parent::new(first.state(AUTH), &first);
        let edited = parse(
            r#"{"model":"m","messages":[{"role":"user","content":"z"},{"role":"user","content":"b"}]}"#,
        );
        assert!(parent.delta(&edited).is_none());
        let other_list = parse(r#"{"model":"m","input":[{"role":"user","content":"a"}]}"#);
        assert!(parent.delta(&other_list).is_none());
    }

    #[test]
    fn only_objects_with_a_list_are_requests() {
        for body in [
            r#"{"model":"m","input":"hi"}"#,
            r#"{"model":"m","prompt":"hi"}"#,
            r#"[{"role":"user"}]"#,
            "not json",
        ] {
            assert!(Request::parse(body.as_bytes()).is_none(), "{body}");
        }
    }

    #[test]
    fn digests_and_headers_round_trip() {
        let request = parse(r#"{"model":"m","messages":[]}"#);
        let digest = request.state(None);
        assert_eq!(digest.to_string().parse::<Digest>(), Ok(digest));
        assert_eq!("00".parse::<Digest>(), Err(Malformed));
        assert_eq!("parent=00; keep=1".parse::<Delta>(), Err(Malformed));
        // Golden, so a change to the digests is deliberate.
        assert_eq!(digest.to_string(), "208b5b1678a0e8f9e5dbca8b4f38d684");
    }
}
