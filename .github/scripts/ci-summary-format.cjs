// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use strict';

const MARKER = '<!-- dynamo-ci-summary -->';
const MAX_BODY = 60000;
const MAX_EXCERPT = 2000;
const MAX_FAILURES = 20;
const FAILURE_CONCLUSIONS = new Set(['failure', 'timed_out', 'action_required', 'startup_failure']);

function sanitize(value) {
  return String(value ?? '')
    .replace(/\x1b\][^\x07]*(?:\x07|\x1b\\)/g, '')
    .replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, '')
    .replace(/\r\n?/g, '\n')
    .replace(/[\u0000-\u0008\u000b-\u001f\u007f-\u009f\u202a-\u202e\u2066-\u2069]/g, '')
    .replace(/\t/g, '  ')
    .replace(/\b(?:hf_|gh[pousr]_|github_pat_)[A-Za-z0-9_]+\b/g, '[REDACTED]')
    .replace(/\b(authorization\s*["']?\s*[:=]\s*["']?)(?:bearer|basic)\s+[^\s"',;]+/gi, '$1[REDACTED]')
    .replace(/\b([\w.-]*(?:password|passwd|token|secret|api[_-]?key|access[_-]?key|auth(?:orization)?)[\w.-]*["']?\s*[:=]\s*)(?:"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'|[^\s,;]+)/gi, '$1[REDACTED]')
    .replace(/(https?:\/\/)[^\s/@:]+:[^\s/@]+@/gi, '$1[REDACTED]@')
    .replace(/@(?!\u200b)/g, '@\u200b');
}

function failureScore(line) {
  if (/^FAILED\s+\S+|^ERROR\s+(?:collecting|at (?:setup|teardown)|\S*(?:\.py|::))/i.test(line)) return 120;
  if (/thread .* panicked at|\bassertion .*failed|AssertionError|^error(?:\[[^\]]+\])?:|fatal error:/i.test(line)) return 100;
  if (/^E\s+|\b\w+Error:|\b(?:left|right|expected|actual):|\b(?:FAILED|FAIL)\b/i.test(line)) return 80;
  if (/\berror\b|\bexception\b|Traceback|ninja: build stopped|make(?:\[\d+\])?: \*\*\*/i.test(line)
      && !/exit(?:ed)?(?: with)? (?:code|status)|process completed/i.test(line)) return 60;
  if (/exit(?:ed)?(?: with)? (?:code|status)|process completed/i.test(line)) return 10;
  return 0;
}

function clip(value, limit) {
  if (value.length <= limit) return value;
  const marker = ' [... truncated ...] ';
  const kept = limit - marker.length;
  const start = Math.ceil(kept / 2);
  return `${value.slice(0, start)}${marker}${value.slice(-(kept - start))}`;
}

function extractFailureExcerpt(log) {
  let raw = String(log ?? '');
  // Keep both ends, dropping partial boundary lines so truncated secrets are not exposed.
  if (raw.length > 2000000) {
    const start = raw.slice(0, 1000000);
    const end = raw.slice(-1000000);
    raw = `${start.slice(0, start.lastIndexOf('\n'))}\n[... log truncated ...]\n${end.slice(end.indexOf('\n') + 1)}`;
  }
  const lines = sanitize(raw).split('\n')
    .map(line => line.replace(/^\s*\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\s*/, '').trimEnd())
    .filter(line => line.trim());
  if (!lines.length) return '';
  const ranked = lines.map((line, index) => ({index, score: failureScore(line)}))
    .filter(item => item.score > 0)
    .sort((a, b) => b.score - a.score || b.index - a.index);
  const useful = ranked.filter(item => item.score >= (ranked[0]?.score > 10 ? 60 : 10));
  const selected = new Map();
  let used = 0;
  function addLine(index) {
    if (selected.has(index) || index < 0 || index >= lines.length || selected.size >= 12) return;
    const remaining = MAX_EXCERPT - used - (selected.size ? 1 : 0);
    if (remaining < Math.min(lines[index].length, 80)) return;
    const line = clip(lines[index], Math.min(600, remaining));
    used += line.length + (selected.size ? 1 : 0);
    selected.set(index, line);
  }
  // Allocate space to the strongest diagnostics before adding surrounding context.
  for (const {index} of useful.slice(0, 8)) addLine(index);
  for (const {index} of useful.slice(0, 8)) {
    addLine(index + 1);
    addLine(index - 1);
  }
  if (!selected.size) for (let i = lines.length - 1; i >= Math.max(0, lines.length - 12); i--) addLine(i);
  return [...selected].sort(([a], [b]) => a - b).map(([, line]) => line).join('\n');
}

function html(value) {
  return String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function text(value, limit = 160) {
  const plain = sanitize(value).replace(/\s+/g, ' ');
  return (plain.length > limit ? `${plain.slice(0, limit - 1)}…` : plain)
    .replace(/[&<>"'\\`*_[\]{}()#+.!|~-]/g, char => `&#${char.charCodeAt(0)};`);
}

function safeUrl(value) {
  if (typeof value !== 'string' || value.length > 1200 || /[\s<>"'`\\]/.test(value)) return '';
  try {
    const url = new URL(value);
    if (url.protocol !== 'https:' || url.username || url.password) return '';
    return url.href.replace(/[()]/g, char => `%${char.charCodeAt(0).toString(16)}`);
  } catch {
    return '';
  }
}

function link(label, value) {
  const url = safeUrl(value);
  return url ? `[${text(label)}](${url})` : text(label);
}

function counts(jobs) {
  const result = {passed: 0, failed: 0, skipped: 0, cancelled: 0, running: 0, queued: 0, other: 0};
  for (const job of jobs) {
    if (FAILURE_CONCLUSIONS.has(job.conclusion)) result.failed++;
    else if (job.conclusion === 'success') result.passed++;
    else if (job.conclusion === 'skipped') result.skipped++;
    else if (job.conclusion === 'cancelled') result.cancelled++;
    else if (job.status === 'in_progress') result.running++;
    else if (['queued', 'waiting', 'pending', 'requested'].includes(job.status)) result.queued++;
    else result.other++;
  }
  return `${jobs.length} total: ${Object.entries(result).filter(([key, count]) => count || ['passed', 'failed', 'skipped', 'cancelled'].includes(key)).map(([key, count]) => `${count} ${key}`).join(', ')}`;
}

function missingStatus(name) {
  if (name === 'PR') return 'No run for this head; full CI requires approval';
  if (name === 'PR-XPU') return 'Optional / not run';
  return 'Not run';
}

function failureDetails(failure, run) {
  const job = failure.job;
  const jobUrl = safeUrl(job.html_url) || safeUrl(run.html_url);
  const steps = (job.steps || []).filter(step => FAILURE_CONCLUSIONS.has(step.conclusion));
  const stepLines = steps.slice(0, 8).map(step => {
    const number = Number(step.number);
    const url = jobUrl && Number.isSafeInteger(number) && number > 0 ? `${jobUrl}#step:${number}:1` : jobUrl;
    return `- ${link(step.name || 'Unnamed step', url)} (${text(step.conclusion)})`;
  });
  if (steps.length > 8) stepLines.push(`- ${steps.length - 8} more failed steps; see the job log.`);
  const excerpt = clip(sanitize(failure.excerpt), MAX_EXCERPT);
  const unavailable = typeof failure.unavailable === 'string' ? ` (${text(failure.unavailable, 240)})` : '';
  const evidence = excerpt ? `<pre>${html(excerpt)}</pre>` : `Log excerpt unavailable${unavailable}; open the job log for details.`;
  return `<details>\n<summary>${text(job.name || 'Unnamed job')} — ${text(job.conclusion || 'failure')}</summary>\n\n${link('Open job log', jobUrl)}\n\n${stepLines.length ? `Failed steps:\n\n${stepLines.join('\n')}\n\n` : ''}${evidence}\n\n</details>\n\n`;
}

function failuresFor(workflow) {
  const failures = workflow.failures || [];
  const key = job => String(job.id ?? job.name);
  const byJob = new Map(failures.filter(item => item.job).map(item => [key(item.job), item]));
  for (const job of workflow.jobs || []) {
    if (FAILURE_CONCLUSIONS.has(job.conclusion) && !byJob.has(key(job))) byJob.set(key(job), {job});
  }
  return [...byJob.values()];
}

function renderSummary({pullRequest, workflows}) {
  const selected = [];
  const sha = pullRequest.head?.sha || pullRequest.head_sha || 'unavailable';
  let body = `${MARKER}\n## CI status\n\nPR head: <code>${text(sha, 80)}</code>\n\n`;
  body += '| Workflow | Status | Jobs | Run |\n| --- | --- | --- | --- |\n';
  for (const workflow of workflows.slice(0, 20)) {
    const run = workflow.run;
    const status = run ? run.conclusion || run.status || 'Unknown' : missingStatus(workflow.name);
    const runLink = run ? link(`Run ${run.id ?? ''} · attempt ${run.run_attempt ?? 1}`, run.html_url) : '—';
    const row = `| ${text(workflow.name)} | ${text(status)} | ${run ? counts(workflow.jobs || []) : '—'} | ${runLink} |\n`;
    if (body.length + row.length > 18000) break;
    body += row;
    selected.push(workflow);
  }
  if (workflows.length > selected.length) body += `\n${workflows.length - selected.length} additional workflows omitted.\n`;
  body += '\nCounts include skipped and cancelled jobs; they are not passing tests.\n\n';
  const footer = 'Failure excerpts are selected from job logs and may be incomplete. Open the linked job or run for the full logs.\n';
  const groups = selected.filter(workflow => workflow.run).map(workflow => {
    const failures = failuresFor(workflow);
    return {
      workflow, failures,
      heading: `### ${text(workflow.name)} failures\n\n`,
      omission: `${failures.length} additional failed job(s) omitted from this summary. ${link('View the full run', workflow.run.html_url)}.\n\n`,
    };
  }).filter(group => group.failures.length);
  let reserved = footer.length + groups.reduce((total, group) => total + group.heading.length + group.omission.length, 0);
  let detailed = 0;
  for (const {workflow, failures, heading, omission} of groups) {
    reserved -= heading.length;
    let details = '';
    let shown = 0;
    for (const failure of failures) {
      if (detailed >= MAX_FAILURES) break;
      const entry = failureDetails(failure, workflow.run);
      // Preserve room for every remaining omission notice and full-run link.
      if (body.length + heading.length + details.length + entry.length + reserved > MAX_BODY) break;
      details += entry;
      shown++;
      detailed++;
    }
    body += heading + details;
    if (shown < failures.length) body += omission.replace(/^\d+/, String(failures.length - shown));
    reserved -= omission.length;
  }
  body += footer;
  return body;
}

module.exports = {extractFailureExcerpt, renderSummary};
