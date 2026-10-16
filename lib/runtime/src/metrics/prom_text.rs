// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parse Prometheus exposition text into [`TypedFamily`] values.
//!
//! Engines serve their metrics over HTTP in one of two text dialects: the
//! classic format (`text/plain; version=0.0.4`) or OpenMetrics 1.0. Both read
//! into the same typed structure that Python's `prometheus_client` hands over,
//! so [`build_families`](super::prom_typed::build_families) stays the single
//! builder behind the `/metrics` re-export and the OTLP export.
//!
//! Family names follow the typed convention: a counter family is named without
//! `_total` and its samples carry the suffix. Classic text names the family
//! `foo_total`; it is normalised here so both dialects build identically.
//!
//! The parser is strict about sample lines and lenient about comments. A line
//! it cannot read as a sample is an error, because a half-read scrape would
//! silently drop series; a comment it does not know, including `# EOF`, is
//! skipped. Exemplars are dropped.
//!
//! Names may use the UTF-8 quoted syntax (`{"a.b",c="d"} 1`, `# TYPE "a.b"
//! gauge`), which servers emit when the scraper accepts
//! `escaping=allow-utf-8`. Every name is mapped to the legacy character set
//! with the `underscores` scheme, so the families fit the `prometheus` crate
//! and the OTLP exporter unchanged; legacy-valid names pass through as is.

use std::collections::{BTreeMap, HashMap};
use std::iter::Peekable;
use std::str::CharIndices;

use anyhow::{Context, bail};

use super::prom_typed::{TypedFamily, TypedSample};

/// Which exposition dialect a body uses. They differ in counter naming,
/// timestamp units, and the `# EOF` trailer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextFormat {
    /// `text/plain; version=0.0.4`. Timestamps are integer milliseconds.
    Classic,
    /// `application/openmetrics-text`. Timestamps are seconds and the body
    /// ends with `# EOF`.
    OpenMetrics,
}

impl TextFormat {
    /// Pick the dialect from the response `Content-Type`, falling back to the
    /// `# EOF` trailer when the header is missing or generic.
    pub fn detect(content_type: Option<&str>, text: &str) -> Self {
        if let Some(value) = content_type {
            let value = value.trim_start().to_ascii_lowercase();
            if value.starts_with("application/openmetrics-text") {
                return Self::OpenMetrics;
            }
            if value.starts_with("text/plain") {
                return Self::Classic;
            }
        }
        match text.lines().rev().find(|line| !line.trim().is_empty()) {
            Some(last) if last.trim() == "# EOF" => Self::OpenMetrics,
            _ => Self::Classic,
        }
    }
}

/// Parse one exposition body. Families are returned in first-seen order with
/// counter families named bare, so a classic `foo_total` and an OpenMetrics
/// `foo` both arrive as `foo` with `foo_total` samples.
pub fn parse_text(text: &str, format: TextFormat) -> anyhow::Result<Vec<TypedFamily>> {
    let mut parser = Parser {
        format,
        families: Vec::new(),
        index: HashMap::new(),
        current: None,
    };
    for (number, line) in text.lines().enumerate() {
        parser
            .line(line)
            .with_context(|| format!("exposition line {}", number + 1))?;
    }
    Ok(parser.families)
}

struct Parser {
    format: TextFormat,
    families: Vec<TypedFamily>,
    index: HashMap<String, usize>,
    /// The family the most recent metadata or sample line belonged to.
    /// Samples are contiguous within a family, so this resolves most lines
    /// without a lookup.
    current: Option<usize>,
}

impl Parser {
    fn line(&mut self, line: &str) -> anyhow::Result<()> {
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() {
            return Ok(());
        }
        match line.strip_prefix('#') {
            Some(comment) => {
                self.comment(comment);
                Ok(())
            }
            None => self.sample(line),
        }
    }

    fn comment(&mut self, comment: &str) {
        let comment = comment.trim_start_matches(' ');
        let Some((keyword, rest)) = comment.split_once(' ') else {
            // `# EOF` and free-text comments.
            return;
        };
        let Some((name, remainder)) = metadata_name(rest) else {
            return;
        };
        let name = &legacy_metric_name(&name);
        match keyword {
            "HELP" => {
                let index = self.family_for_metadata(name);
                self.families[index].help = unescape(remainder);
                self.current = Some(index);
            }
            "TYPE" => {
                let kind = normalise_kind(remainder.trim());
                let index = self.family_for_type(name, &kind);
                self.families[index].kind = kind;
                self.current = Some(index);
            }
            "UNIT" => {
                let index = self.family_for_metadata(name);
                self.families[index].unit = remainder.trim().to_string();
                self.current = Some(index);
            }
            _ => {}
        }
    }

    fn sample(&mut self, line: &str) -> anyhow::Result<()> {
        let (name, labels, rest) = if line.starts_with('{') {
            // UTF-8 syntax: the quoted metric name leads the label set.
            let (name, labels, rest) = parse_labels(line, true)?;
            (name.context("sample has no metric name")?, labels, rest)
        } else {
            let name_end = line
                .find(|c: char| c == '{' || c.is_whitespace())
                .unwrap_or(line.len());
            let (name, rest) = line.split_at(name_end);
            if name.is_empty() {
                bail!("sample has no metric name");
            }
            let (labels, rest) = if rest.starts_with('{') {
                let (_, labels, rest) = parse_labels(rest, false)?;
                (labels, rest)
            } else {
                (BTreeMap::new(), rest)
            };
            (legacy_metric_name(name), labels, rest)
        };
        let name = name.as_str();
        let mut tokens = rest.split_whitespace();
        let value = tokens.next().context("sample has no value")?;
        let value: f64 = value
            .parse()
            .with_context(|| format!("invalid sample value `{value}`"))?;
        // Anything from an exemplar marker onward is dropped; the text and
        // proto models used downstream do not carry exemplars.
        let timestamp = match tokens.next() {
            None | Some("#") => None,
            Some(raw) => Some(parse_timestamp(raw, self.format)?),
        };
        let index = self.family_for_sample(name);
        self.families[index].samples.push(TypedSample {
            name: name.to_string(),
            labels,
            value,
            timestamp,
        });
        Ok(())
    }

    /// The family a `# TYPE` line declares. Counter families drop a `_total`
    /// suffix so classic text matches the typed naming; a family already
    /// created under the suffixed name by an earlier `# HELP` is renamed.
    fn family_for_type(&mut self, name: &str, kind: &str) -> usize {
        let key = match kind {
            "counter" => name.strip_suffix("_total").unwrap_or(name),
            _ => name,
        };
        if let Some(&index) = self.index.get(key) {
            return index;
        }
        if key != name
            && let Some(index) = self.index.remove(name)
        {
            self.families[index].name = key.to_string();
            self.index.insert(key.to_string(), index);
            return index;
        }
        self.insert(key)
    }

    /// The family a `# HELP` or `# UNIT` line refers to. Classic text names a
    /// counter's metadata `foo_total`; when `foo` is already a counter, that is
    /// the family meant.
    fn family_for_metadata(&mut self, name: &str) -> usize {
        if let Some(&index) = self.index.get(name) {
            return index;
        }
        if let Some(bare) = name.strip_suffix("_total")
            && let Some(&index) = self.index.get(bare)
            && self.families[index].kind == "counter"
        {
            return index;
        }
        self.insert(name)
    }

    fn family_for_sample(&mut self, name: &str) -> usize {
        if let Some(index) = self.current
            && belongs(&self.families[index], name)
        {
            return index;
        }
        if let Some(&index) = self.index.get(name) {
            self.current = Some(index);
            return index;
        }
        if let Some(bare) = name.strip_suffix("_total")
            && let Some(&index) = self.index.get(bare)
            && self.families[index].kind == "counter"
        {
            self.current = Some(index);
            return index;
        }
        // A sample with no metadata at all is untyped.
        let index = self.insert(name);
        self.current = Some(index);
        index
    }

    fn insert(&mut self, name: &str) -> usize {
        let index = self.families.len();
        self.families.push(TypedFamily {
            name: name.to_string(),
            help: String::new(),
            kind: "unknown".to_string(),
            unit: String::new(),
            samples: Vec::new(),
        });
        self.index.insert(name.to_string(), index);
        index
    }
}

/// Whether `sample` is one of `family`'s samples under the family's type.
fn belongs(family: &TypedFamily, sample: &str) -> bool {
    let Some(suffix) = sample.strip_prefix(family.name.as_str()) else {
        return false;
    };
    match family.kind.as_str() {
        "counter" => matches!(suffix, "" | "_total" | "_created"),
        "histogram" => matches!(suffix, "_bucket" | "_sum" | "_count" | "_created"),
        "gaugehistogram" => matches!(suffix, "_bucket" | "_gsum" | "_gcount"),
        "summary" => matches!(suffix, "" | "_sum" | "_count" | "_created"),
        "info" => suffix == "_info",
        _ => suffix.is_empty(),
    }
}

fn normalise_kind(kind: &str) -> String {
    match kind.to_ascii_lowercase().as_str() {
        // The classic word for an untyped family; the typed builder knows `unknown`.
        "untyped" | "" => "unknown".to_string(),
        other => other.to_string(),
    }
}

fn parse_timestamp(raw: &str, format: TextFormat) -> anyhow::Result<f64> {
    let number: f64 = raw
        .parse()
        .with_context(|| format!("invalid sample timestamp `{raw}`"))?;
    Ok(match format {
        TextFormat::Classic => number / 1_000.0,
        TextFormat::OpenMetrics => number,
    })
}

/// Parse `{name="value",...}` at the start of `text`, returning the labels and
/// the remainder after the closing brace. With `leading_name`, a quoted first
/// item that is not followed by `=` is the metric name (UTF-8 syntax); it is
/// returned already mapped to the legacy form, as are the label names.
fn parse_labels(
    text: &str,
    leading_name: bool,
) -> anyhow::Result<(Option<String>, BTreeMap<String, String>, &str)> {
    let mut metric = None;
    let mut labels = BTreeMap::new();
    let mut chars = text.char_indices().peekable();
    // Consume the opening brace.
    chars.next();
    let mut first = true;
    loop {
        // Skip whitespace and separators between pairs.
        while chars
            .peek()
            .is_some_and(|(_, c)| c.is_whitespace() || *c == ',')
        {
            chars.next();
        }
        let Some(&(position, c)) = chars.peek() else {
            bail!("labels are not closed");
        };
        if c == '}' {
            return Ok((metric, labels, &text[position + 1..]));
        }
        let name = if c == '"' {
            let quoted = read_quoted(&mut chars, "quoted name")?;
            while chars.peek().is_some_and(|(_, c)| c.is_whitespace()) {
                chars.next();
            }
            if first && leading_name && chars.peek().is_some_and(|(_, c)| matches!(c, ',' | '}')) {
                metric = Some(legacy_metric_name(&quoted));
                first = false;
                continue;
            }
            legacy_label_name(&quoted)
        } else {
            let name_start = position;
            let mut name_end = name_start;
            while let Some(&(position, c)) = chars.peek() {
                if c == '=' {
                    break;
                }
                if c.is_whitespace() || c == ',' || c == '}' || c == '"' {
                    bail!("label name is not followed by `=`");
                }
                chars.next();
                name_end = position + c.len_utf8();
            }
            legacy_label_name(&text[name_start..name_end])
        };
        first = false;
        if name.is_empty() {
            bail!("empty label name");
        }
        if chars.next().map(|(_, c)| c) != Some('=') {
            bail!("label `{name}` has no value");
        }
        if chars.peek().map(|(_, c)| *c) != Some('"') {
            bail!("label `{name}` value is not quoted");
        }
        let value = read_quoted(&mut chars, &format!("label `{name}` value"))?;
        labels.insert(name, value);
    }
}

/// Read a `"..."` string whose opening quote is the next character, undoing
/// the `\"`, `\\`, and `\n` escapes both dialects use.
fn read_quoted(chars: &mut Peekable<CharIndices<'_>>, what: &str) -> anyhow::Result<String> {
    if chars.next().map(|(_, c)| c) != Some('"') {
        bail!("{what} is not quoted");
    }
    let mut value = String::new();
    loop {
        let Some((_, c)) = chars.next() else {
            bail!("{what} is not closed");
        };
        match c {
            '"' => return Ok(value),
            '\\' => match chars.next().map(|(_, c)| c) {
                Some('n') => value.push('\n'),
                Some('"') => value.push('"'),
                Some('\\') => value.push('\\'),
                Some(other) => {
                    value.push('\\');
                    value.push(other);
                }
                None => bail!("{what} ends in a backslash"),
            },
            other => value.push(other),
        }
    }
}

/// Split the name off a `# HELP`/`# TYPE`/`# UNIT` payload. The name is bare
/// or, in the UTF-8 syntax, quoted; the remainder follows one space.
fn metadata_name(rest: &str) -> Option<(String, &str)> {
    if rest.starts_with('"') {
        let mut chars = rest.char_indices().peekable();
        let name = read_quoted(&mut chars, "metadata name").ok()?;
        let remainder = chars
            .next()
            .map_or("", |(position, _)| rest[position..].trim_start_matches(' '));
        return (!name.is_empty()).then_some((name, remainder));
    }
    let (name, remainder) = rest
        .split_once(' ')
        .map_or((rest, ""), |(name, remainder)| (name, remainder));
    (!name.is_empty()).then(|| (name.to_string(), remainder))
}

/// The Prometheus `underscores` escaping scheme for metric names: every
/// character outside `[a-zA-Z0-9_:]` becomes `_`, and a leading digit gains a
/// `_` prefix. Legacy-valid names are unchanged.
fn legacy_metric_name(name: &str) -> String {
    legacy_name(name, true)
}

/// The same scheme for label names, where `:` is not legacy-valid either.
fn legacy_label_name(name: &str) -> String {
    legacy_name(name, false)
}

fn legacy_name(name: &str, allow_colon: bool) -> String {
    let valid = |c: char| c.is_ascii_alphanumeric() || c == '_' || (allow_colon && c == ':');
    if name.chars().all(valid) && !name.starts_with(|c: char| c.is_ascii_digit()) {
        return name.to_string();
    }
    let mut out = String::with_capacity(name.len() + 1);
    if name.starts_with(|c: char| c.is_ascii_digit()) {
        out.push('_');
    }
    out.extend(name.chars().map(|c| if valid(c) { c } else { '_' }));
    out
}

/// Undo the escaping both dialects apply to help text.
fn unescape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('"') => out.push('"'),
            Some('\\') => out.push('\\'),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn family<'a>(families: &'a [TypedFamily], name: &str) -> &'a TypedFamily {
        families
            .iter()
            .find(|family| family.name == name)
            .unwrap_or_else(|| panic!("family {name} missing from {families:?}"))
    }

    #[test]
    fn classic_counter_is_named_bare_with_suffixed_samples() {
        let text = "\
# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name=\"Qwen/Qwen3-0.6B\"} 19132.0
# HELP vllm:prompt_tokens_created Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_created gauge
vllm:prompt_tokens_created{model_name=\"Qwen/Qwen3-0.6B\"} 1.7e9
";
        let families = parse_text(text, TextFormat::Classic).unwrap();
        assert_eq!(families.len(), 2);
        let counter = family(&families, "vllm:prompt_tokens");
        assert_eq!(counter.kind, "counter");
        assert_eq!(counter.help, "Number of prefill tokens processed.");
        assert_eq!(counter.samples.len(), 1);
        assert_eq!(counter.samples[0].name, "vllm:prompt_tokens_total");
        assert_eq!(counter.samples[0].value, 19132.0);
        assert_eq!(
            counter.samples[0].labels.get("model_name").unwrap(),
            "Qwen/Qwen3-0.6B"
        );
        assert_eq!(
            family(&families, "vllm:prompt_tokens_created").kind,
            "gauge"
        );
    }

    #[test]
    fn classic_histogram_and_summary_group_their_samples() {
        let text = "\
# HELP ttft Time to first token.
# TYPE ttft histogram
ttft_bucket{le=\"0.01\",engine=\"0\"} 27
ttft_bucket{le=\"+Inf\",engine=\"0\"} 138
ttft_sum{engine=\"0\"} 1.5065741539001465
ttft_count{engine=\"0\"} 138
# TYPE latency summary
latency{quantile=\"0.5\"} 0.2
latency{quantile=\"0.99\"} 0.9
latency_sum 12
latency_count 40
";
        let families = parse_text(text, TextFormat::Classic).unwrap();
        assert_eq!(families.len(), 2);
        let histogram = family(&families, "ttft");
        assert_eq!(histogram.kind, "histogram");
        assert_eq!(histogram.samples.len(), 4);
        assert_eq!(histogram.samples[1].labels.get("le").unwrap(), "+Inf");
        let summary = family(&families, "latency");
        assert_eq!(summary.kind, "summary");
        assert_eq!(summary.samples.len(), 4);
        assert_eq!(summary.samples[0].labels.get("quantile").unwrap(), "0.5");
    }

    #[test]
    fn utf8_quoted_names_are_mapped_to_the_underscores_form() {
        // What prometheus_client 0.26 serves for `escaping=allow-utf-8`: legacy
        // names verbatim, anything else quoted.
        let text = "\
# HELP sglang:test_gauge h
# TYPE sglang:test_gauge gauge
sglang:test_gauge{model_name=\"m\"} 1.0
# HELP \"dotted.name\" dotted help
# TYPE \"dotted.name\" gauge
{\"dotted.name\"} 2.0
# HELP \"label.dots\" h
# TYPE \"label.dots\" counter
{\"label.dots_total\",\"a.b\"=\"v\"} 3.0
{\"label.dots_created\",\"a.b\"=\"v\"} 1.7e9
# TYPE \"q\\\"x\" gauge
{ \"q\\\"x\" , z=\"2\" } 4
plain{\"x.y\"=\"1\",z=\"2\"} 5
# EOF
";
        let families = parse_text(text, TextFormat::OpenMetrics).unwrap();
        assert_eq!(families.len(), 5, "{families:?}");
        let colon = family(&families, "sglang:test_gauge");
        assert_eq!(colon.samples[0].name, "sglang:test_gauge");
        let dotted = family(&families, "dotted_name");
        assert_eq!(dotted.help, "dotted help");
        assert_eq!(dotted.samples[0].name, "dotted_name");
        assert_eq!(dotted.samples[0].value, 2.0);
        let counter = family(&families, "label_dots");
        assert_eq!(counter.kind, "counter");
        assert_eq!(counter.samples.len(), 2);
        assert_eq!(counter.samples[0].name, "label_dots_total");
        assert_eq!(counter.samples[0].labels.get("a_b").unwrap(), "v");
        let quoted = family(&families, "q_x");
        assert_eq!(quoted.samples[0].labels.get("z").unwrap(), "2");
        assert_eq!(quoted.samples[0].value, 4.0);
        let plain = family(&families, "plain");
        assert_eq!(plain.samples[0].labels.get("x_y").unwrap(), "1");
        assert_eq!(legacy_metric_name("9lives"), "_9lives");
        assert_eq!(legacy_label_name("a:b"), "a_b");
    }

    #[test]
    fn mismatched_metadata_and_sample_names_leave_the_family_empty() {
        // prometheus_client 0.26 without `escaping=allow-utf-8`
        // (prometheus/client_python#1177): the samples become their own
        // untyped families, so a `sglang:` prefix filter keeps nothing.
        let text = "\
# HELP sglang:x h
# TYPE sglang:x gauge
sglang_x 1
# EOF
";
        let families = parse_text(text, TextFormat::OpenMetrics).unwrap();
        assert!(family(&families, "sglang:x").samples.is_empty());
        assert_eq!(family(&families, "sglang_x").kind, "unknown");
    }

    #[test]
    fn samples_without_metadata_are_untyped_and_timestamps_convert() {
        let text = "orphan{a=\"b\"} 3 1700000000123\nplain 1\n";
        let families = parse_text(text, TextFormat::Classic).unwrap();
        assert_eq!(families.len(), 2);
        let orphan = family(&families, "orphan");
        assert_eq!(orphan.kind, "unknown");
        assert_eq!(orphan.samples[0].timestamp, Some(1_700_000_000.123));
        assert_eq!(family(&families, "plain").samples[0].value, 1.0);

        let families = parse_text("ts 2 1700000000.5\n", TextFormat::OpenMetrics).unwrap();
        assert_eq!(families[0].samples[0].timestamp, Some(1_700_000_000.5));
    }

    #[test]
    fn labels_handle_escapes_commas_and_trailing_separators() {
        let text = "m{path=\"a,b\",quote=\"say \\\"hi\\\"\",nl=\"x\\ny\",back=\"c:\\\\d\",} +Inf\n";
        let families = parse_text(text, TextFormat::Classic).unwrap();
        let sample = &families[0].samples[0];
        assert_eq!(sample.labels.get("path").unwrap(), "a,b");
        assert_eq!(sample.labels.get("quote").unwrap(), "say \"hi\"");
        assert_eq!(sample.labels.get("nl").unwrap(), "x\ny");
        assert_eq!(sample.labels.get("back").unwrap(), "c:\\d");
        assert!(sample.value.is_infinite() && sample.value > 0.0);
    }

    #[test]
    fn help_text_is_unescaped() {
        let text = "# HELP m First line\\nsecond \\\\ line\n# TYPE m gauge\nm 1\n";
        let families = parse_text(text, TextFormat::Classic).unwrap();
        assert_eq!(families[0].help, "First line\nsecond \\ line");
    }

    #[test]
    fn openmetrics_counters_info_units_and_exemplars() {
        let text = "\
# HELP requests Requests served.
# TYPE requests counter
# UNIT requests requests
requests_total{code=\"200\"} 42 # {trace_id=\"abc\"} 1 1700000000
requests_created{code=\"200\"} 1700000000
# TYPE build info
build_info{version=\"1.0\"} 1
# TYPE state stateset
state{state=\"ready\"} 1
# EOF
";
        let families = parse_text(text, TextFormat::OpenMetrics).unwrap();
        assert_eq!(families.len(), 3);
        let requests = family(&families, "requests");
        assert_eq!(requests.kind, "counter");
        assert_eq!(requests.unit, "requests");
        assert_eq!(requests.samples.len(), 2);
        assert_eq!(requests.samples[0].value, 42.0);
        assert_eq!(requests.samples[0].timestamp, None);
        assert_eq!(requests.samples[1].name, "requests_created");
        let build = family(&families, "build");
        assert_eq!(build.kind, "info");
        assert_eq!(build.samples[0].name, "build_info");
        assert_eq!(family(&families, "state").kind, "stateset");
    }

    #[test]
    fn adjacent_families_with_shared_prefixes_do_not_merge() {
        let text = "\
# TYPE foo gauge
foo 1
# TYPE foo_bar gauge
foo_bar 2
foo_baz 3
";
        let families = parse_text(text, TextFormat::Classic).unwrap();
        let names: Vec<&str> = families.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, ["foo", "foo_bar", "foo_baz"]);
        assert_eq!(family(&families, "foo_baz").kind, "unknown");
    }

    #[test]
    fn malformed_samples_fail_with_the_line_number() {
        for (text, needle) in [
            ("ok 1\nbad{x=\"1\" 2\n", "line 2"),
            ("novalue\n", "no value"),
            ("m{a=b} 1\n", "not quoted"),
            ("m nan_ 1\n", "invalid sample value"),
        ] {
            let error = parse_text(text, TextFormat::Classic).unwrap_err();
            assert!(format!("{error:#}").contains(needle), "{text:?}: {error:#}");
        }
    }

    #[test]
    fn detect_prefers_content_type_then_trailer() {
        assert_eq!(
            TextFormat::detect(
                Some("application/openmetrics-text; version=1.0.0; charset=utf-8"),
                ""
            ),
            TextFormat::OpenMetrics
        );
        assert_eq!(
            TextFormat::detect(Some("text/plain; version=0.0.4"), "m 1\n# EOF\n"),
            TextFormat::Classic
        );
        assert_eq!(
            TextFormat::detect(None, "m 1\n# EOF\n\n"),
            TextFormat::OpenMetrics
        );
        assert_eq!(TextFormat::detect(None, "m 1\n"), TextFormat::Classic);
    }

    /// Lines captured from vLLM's Rust frontend, which renders OpenMetrics.
    #[test]
    fn vllm_openmetrics_excerpt_parses() {
        let text = "\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name=\"Qwen/Qwen3-0.6B\",engine=\"0\"} 0
# HELP vllm:prompt_tokens Number of prefill tokens processed.
# TYPE vllm:prompt_tokens counter
vllm:prompt_tokens_total{model_name=\"Qwen/Qwen3-0.6B\",engine=\"0\"} 19132
# HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_sum{model_name=\"Qwen/Qwen3-0.6B\",engine=\"0\"} 1.5065741539001465
vllm:time_to_first_token_seconds_count{model_name=\"Qwen/Qwen3-0.6B\",engine=\"0\"} 138
vllm:time_to_first_token_seconds_bucket{le=\"0.001\",model_name=\"Qwen/Qwen3-0.6B\",engine=\"0\"} 0
vllm:time_to_first_token_seconds_bucket{le=\"+Inf\",model_name=\"Qwen/Qwen3-0.6B\",engine=\"0\"} 138
http_requests_total{handler=\"/metrics\",method=\"GET\",status=\"2xx\"} 5
# EOF
";
        let families = parse_text(text, TextFormat::OpenMetrics).unwrap();
        let names: Vec<&str> = families.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "vllm:num_requests_running",
                "vllm:prompt_tokens",
                "vllm:time_to_first_token_seconds",
                "http_requests_total",
            ]
        );
        assert_eq!(family(&families, "vllm:prompt_tokens").kind, "counter");
        assert_eq!(
            family(&families, "vllm:time_to_first_token_seconds")
                .samples
                .len(),
            4
        );
    }
}
