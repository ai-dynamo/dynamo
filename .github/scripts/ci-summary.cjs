// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const { extractFailureExcerpt, renderSummary } = require('./ci-summary-format.cjs');

const WORKFLOWS = [
  { name: 'Pre Merge', file: 'pre-merge.yml', event: 'pull_request' },
  { name: 'PR', file: 'pr.yaml', event: 'push' },
  { name: 'PR-XPU', file: 'pr-xpu.yaml', event: 'push' },
];
const MARKER = '<!-- dynamo-ci-summary -->';
const FAILED = new Set(['failure', 'timed_out', 'action_required', 'startup_failure']);
const MAX_LOGS = 10;
const MAX_LOG_BYTES = 1024 * 1024;

function workflowFor(run) {
  return WORKFLOWS.find(w => run.path === `.github/workflows/${w.file}` && run.event === w.event);
}

function matchesPullRequest(run, pr, repo) {
  const workflow = workflowFor(run);
  if (!workflow || run.head_sha !== pr.head.sha || run.repository.full_name !== repo) return false;
  if (workflow.event === 'push') {
    return run.head_branch === `pull-request/${pr.number}` && run.head_repository?.full_name === repo;
  }
  const associated = !run.pull_requests?.length || run.pull_requests.some(candidate =>
    candidate.number === pr.number && (!candidate.base?.ref || candidate.base.ref === pr.base.ref));
  return associated && run.head_branch === pr.head.ref && run.head_repository?.full_name === pr.head.repo?.full_name;
}

async function resolvePullRequest({ github, context, core }) {
  const { data: run } = await github.rest.actions.getWorkflowRun({
    ...context.repo, run_id: context.payload.workflow_run.id,
  });
  const workflow = workflowFor(run);
  const repo = `${context.repo.owner}/${context.repo.repo}`;
  if (!workflow || run.repository.full_name !== repo) return null;
  let numbers;
  if (workflow.event === 'push') {
    const match = /^pull-request\/([1-9]\d*)$/.exec(run.head_branch || '');
    if (!match) return null;
    numbers = [Number(match[1])];
  } else {
    let associated = run.pull_requests || [];
    // Fork workflow payloads can omit pull_requests; ask GitHub for the association.
    if (!associated.length) {
      associated = await github.paginate(github.rest.repos.listPullRequestsAssociatedWithCommit, {
        ...context.repo, commit_sha: run.head_sha, per_page: 100,
      });
    }
    if (!associated.length && run.head_repository?.full_name && run.head_branch) {
      associated = await github.paginate(github.rest.pulls.list, {
        ...context.repo, state: 'open',
        head: `${run.head_repository.full_name.split('/')[0]}:${run.head_branch}`, per_page: 100,
      });
    }
    numbers = [...new Set(associated.map(pr => pr.number))];
  }
  const matches = [];
  for (const number of numbers) {
    const { data: pr } = await github.rest.pulls.get({ ...context.repo, pull_number: number });
    if (pr.state === 'open' && pr.base.repo.full_name === repo && matchesPullRequest(run, pr, repo)) {
      matches.push(pr);
    }
  }
  if (matches.length !== 1) {
    core.info('No unambiguous open PR at the tested head; skipping the summary.');
    return null;
  }
  return matches[0];
}

async function latestRuns(github, repo, pr) {
  return Promise.all(WORKFLOWS.map(async workflow => {
    const runs = await github.paginate(github.rest.actions.listWorkflowRuns, {
      ...repo, workflow_id: workflow.file, event: workflow.event,
      branch: workflow.event === 'push' ? `pull-request/${pr.number}` : pr.head.ref,
      head_sha: pr.head.sha, per_page: 100,
    });
    const matching = runs.filter(run => matchesPullRequest(run, pr, `${repo.owner}/${repo.repo}`));
    matching.sort((a, b) => b.id - a.id);
    for (const run of matching) {
      if (workflow.event === 'pull_request' && !run.pull_requests?.length) {
        const sourcePRs = await github.paginate(github.rest.pulls.list, {
          ...repo, state: 'open', head: `${pr.head.repo.full_name.split('/')[0]}:${pr.head.ref}`,
          per_page: 100,
        });
        const atHead = sourcePRs.filter(candidate => candidate.head.sha === pr.head.sha &&
          candidate.head.repo?.full_name === pr.head.repo.full_name);
        // Without an association, equal source SHAs cannot distinguish different PR bases.
        if (atHead.length !== 1 || atHead[0].number !== pr.number) continue;
      }
      return { name: workflow.name, run };
    }
    return { name: workflow.name, run: null };
  }));
}

async function readJobLog({ repo, jobId, token, apiUrl = 'https://api.github.com', fetchImpl = fetch }) {
  const response = await fetchImpl(`${apiUrl}/repos/${repo.owner}/${repo.repo}/actions/jobs/${jobId}/logs`, {
    headers: { Authorization: `Bearer ${token}`, Accept: 'application/vnd.github+json' },
    redirect: 'manual', signal: AbortSignal.timeout(15000),
  });
  const location = response.headers.get('location');
  if (response.status !== 302 || !location || new URL(location).protocol !== 'https:') {
    throw new Error('Job log download is unavailable');
  }
  // Do not forward the GitHub token to the signed log-storage URL.
  const log = await fetchImpl(location, {
    headers: { Range: `bytes=-${MAX_LOG_BYTES}` }, signal: AbortSignal.timeout(15000),
  });
  if (!log.ok || !log.body) throw new Error('Job log download failed');
  const reader = log.body.getReader();
  const chunks = [];
  let length = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      length += value.length;
      // Some storage providers ignore Range; never buffer an unbounded CI log.
      if (length > MAX_LOG_BYTES) throw new Error('Job log exceeded the download limit');
      chunks.push(Buffer.from(value));
    }
  } finally {
    await reader.cancel();
  }
  const text = Buffer.concat(chunks).toString('utf8');
  const rangeStart = log.headers?.get('content-range')?.match(/^bytes (\d+)-/);
  // A suffix range may start mid-secret; discard its partial first line before redaction.
  if (log.status === 206 && (!rangeStart || Number(rangeStart[1]) > 0)) {
    const newline = text.indexOf('\n');
    return newline === -1 ? '' : text.slice(newline + 1);
  }
  return text;
}

async function collectSummary({ github, context, pullRequest, core, loadLog }) {
  const workflows = await latestRuns(github, context.repo, pullRequest);
  let remainingLogs = MAX_LOGS;
  for (const workflow of workflows) {
    workflow.jobs = [];
    workflow.failures = [];
    if (!workflow.run) continue;
    // latest retains successful jobs from earlier attempts when only failures rerun.
    workflow.jobs = await github.paginate(github.rest.actions.listJobsForWorkflowRun, {
      ...context.repo, run_id: workflow.run.id, filter: 'latest', per_page: 100,
    });
    for (const job of workflow.jobs.filter(job => FAILED.has(job.conclusion))) {
      let excerpt = '';
      let unavailable = 'No diagnostic excerpt is available; inspect the linked job.';
      if (remainingLogs > 0) {
        remainingLogs--;
        try {
          excerpt = extractFailureExcerpt(await loadLog(job.id));
        } catch {
          // API errors may contain signed URLs; only log the public job ID.
          core.info(`Could not retrieve a bounded log excerpt for job ${job.id}.`);
        }
      } else {
        unavailable = 'Log excerpt limit reached; inspect the linked job.';
      }
      workflow.failures.push({ job, excerpt, unavailable });
    }
  }
  return { pullRequest, workflows };
}

function snapshot(workflows) {
  return workflows.map(({ run }) => run && [run.id, run.run_attempt, run.status, run.conclusion]);
}

async function publishSummary({ github, context, core, pullNumber, loadLog, dryRun = false }) {
  const { data: pr } = await github.rest.pulls.get({ ...context.repo, pull_number: pullNumber });
  const repo = `${context.repo.owner}/${context.repo.repo}`;
  if (pr.state !== 'open' || pr.base.repo.full_name !== repo) return null;
  const summary = await collectSummary({
    github, context, pullRequest: pr, core,
    loadLog: loadLog || (jobId => readJobLog({
      repo: context.repo, jobId, token: process.env.GH_TOKEN, apiUrl: process.env.GITHUB_API_URL,
    })),
  });
  const body = renderSummary(summary);
  if (dryRun) return body;
  const comments = await github.paginate(github.rest.issues.listComments, {
    ...context.repo, issue_number: pr.number, per_page: 100,
  });
  const comment = comments.find(comment => comment.user?.login === 'github-actions[bot]' &&
    comment.user.type === 'Bot' && comment.body?.startsWith(MARKER));
  const { data: current } = await github.rest.pulls.get({ ...context.repo, pull_number: pr.number });
  const freshRuns = await latestRuns(github, context.repo, current);
  if (current.state !== 'open' || current.head.sha !== pr.head.sha ||
      JSON.stringify(snapshot(freshRuns)) !== JSON.stringify(snapshot(summary.workflows))) {
    core.info('PR head or CI runs changed while collecting results; the next event will refresh the comment.');
    return null;
  }
  if (comment?.body === body) return body;
  if (comment) {
    await github.rest.issues.updateComment({ ...context.repo, comment_id: comment.id, body });
  } else {
    await github.rest.issues.createComment({ ...context.repo, issue_number: pr.number, body });
  }
  return body;
}

module.exports = { resolvePullRequest, collectSummary, publishSummary, readJobLog, matchesPullRequest };
