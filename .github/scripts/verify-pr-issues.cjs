// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// API contracts: https://docs.github.com/en/rest/issues/issues#get-an-issue
// https://linear.app/developers/graphql (issue(id:) accepts a human-readable ID).
function references(text, repo) {
  // Template examples and other invisible comments are not references.
  text = text.replace(/<!--[\s\S]*?(?:-->|$)/g, '');
  const found = new Map();
  const github = (owner, name, number) => {
    const ref = `${owner}/${name}#${Number(number)}`;
    found.set(ref.toLowerCase(), {
      kind: 'github', ref, owner, repo: name, number: Number(number),
    });
  };
  text = text.replace(
    /https:\/\/github\.com\/([\w.-]+)\/([\w.-]+)\/issues\/([1-9]\d*)\b[^\s<>)]*/gi,
    (_, owner, name, number) => {
      github(owner, name, number);
      return ' ';
    },
  );
  // Remove other URLs so anchors, pull links and version numbers in URLs
  // cannot accidentally satisfy the requirement.
  text = text.replace(
    /https:\/\/linear\.app\/[\w-]+\/issue\/([A-Z][A-Z0-9]*-[1-9]\d*)\b[^\s<>)]*/gi,
    (_, id) => ` ${id.toUpperCase()} `,
  ).replace(/https?:\/\/[^\s<>)]*/gi, ' ');
  text = text.replace(
    /(?<![\w/.-])([\w.-]+)\/([\w.-]+)#([1-9]\d*)\b/g,
    (_, owner, name, number) => {
      github(owner, name, number);
      return ' ';
    },
  );
  for (const match of text.matchAll(/(?<![\w/])(?:#|GH-)([1-9]\d*)\b/g)) {
    github(repo.owner, repo.repo, match[1]);
  }
  for (const match of text.matchAll(/\b([A-Z][A-Z0-9]*-[1-9]\d*)\b/g)) {
    const id = match[1];
    if (!id.startsWith('GH-')) found.set(id, { kind: 'linear', ref: id });
  }
  return [...found.values()];
}

async function lookupGitHub(ref, github) {
  try {
    const { data } = await github.rest.issues.get({
      owner: ref.owner, repo: ref.repo, issue_number: ref.number,
    });
    // GitHub's issues endpoint also returns pull requests.
    if (data.pull_request) return { error: 'is a pull request, not an issue' };
    return { title: data.title };
  } catch (error) {
    if (error.status === 404) return { error: 'does not exist or is not accessible' };
    // Do not log raw API errors: they may contain request headers or secrets.
    return { error: `could not be verified: GitHub API ${error.status || 'request failure'}` };
  }
}

async function lookupLinear(ref, key, fetchImpl) {
  if (!key) return { error: 'could not be verified: LINEAR_ACCESS_KEY secret is missing' };
  try {
    const response = await fetchImpl('https://api.linear.app/graphql', {
      method: 'POST',
      headers: { Authorization: key, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: 'query IssueReference($id: String!) { issue(id: $id) { identifier title } }',
        variables: { id: ref.ref },
      }),
      signal: AbortSignal.timeout(15000),
      redirect: 'error',
    });
    if (!response.ok) return { error: `could not be verified: Linear API HTTP ${response.status}` };
    const payload = await response.json();
    if (payload.errors?.length) {
      if (payload.errors.every(error => /entity not found/i.test(error.message || ''))) {
        return { error: 'does not exist or is not accessible' };
      }
      return { error: 'could not be verified: Linear GraphQL error (check key access and API availability)' };
    }
    if (payload.data?.issue === null) return { error: 'does not exist or is not accessible' };
    const issue = payload.data?.issue;
    if (!issue || typeof issue.title !== 'string' || !issue.identifier) {
      return { error: 'could not be verified: unexpected Linear API response' };
    }
    return { title: issue.title };
  } catch {
    return { error: 'could not be verified: Linear request failed or timed out' };
  }
}

// Escape API titles and PR input before including them in the HTML summary.
function escape(value) {
  return String(value).replace(/[&<>"']/g, char => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[char]);
}

async function verify({ github, context, core, key = process.env.LINEAR_ACCESS_KEY, fetchImpl = fetch }) {
  const pr = context.payload.pull_request;
  const refs = references(`${pr.title || ''}\n${pr.body || ''}`, context.repo);
  const summary = core.summary.addHeading('PR issue references');
  if (!refs.length) {
    const message = 'Failed: no issue listed. Add a GitHub issue (#123, owner/repo#123, or an issue URL) or Linear ticket (TEAM-123 or a Linear issue URL) to the PR title or description.';
    await summary.addRaw(`<p>${escape(message)}</p>`).write();
    core.setFailed(message);
    return;
  }
  // Bound untrusted input to avoid exhausting API quotas on a single PR.
  if (refs.length > 50) {
    const message = 'Failed: more than 50 issue references listed; reduce the number of references.';
    await summary.addRaw(`<p>${escape(message)}</p>`).write();
    core.setFailed(message);
    return;
  }
  const failures = [];
  const rows = [];
  for (const ref of refs) {
    const result = ref.kind === 'github'
      ? await lookupGitHub(ref, github)
      : await lookupLinear(ref, key, fetchImpl);
    const message = result.error
      ? `Failed: issue listed ${ref.ref} ${result.error}`
      : `Found: ${ref.ref} — ${result.title}`;
    if (result.error) {
      failures.push(message);
      core.error(message);
    } else {
      core.info(message.replace(/[\r\n]/g, ' '));
    }
    rows.push(`<tr><td>${escape(ref.ref)}</td><td>${result.error ? 'Failed' : 'Found'}</td><td>${escape(result.error || result.title)}</td></tr>`);
  }
  summary.addRaw(`<p>${failures.length ? 'Failed' : 'Passed'}: ${refs.length - failures.length} of ${refs.length} referenced issues verified.</p>`);
  await summary.addRaw(`<table><tr><th>Reference</th><th>Result</th><th>Title / reason</th></tr>${rows.join('')}</table>`).write();
  if (failures.length) core.setFailed(failures.join('\n'));
}

module.exports = { references, verify };
