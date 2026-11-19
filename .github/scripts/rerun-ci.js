// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

module.exports = async function rerunFailedCI({ github, context, core }) {
  const WORKFLOWS = new Set(['.github/workflows/pr.yaml', '.github/workflows/pr-xpu.yaml']);
  const COPY_PR_BOT_ID = 138252374;
  const ACTIONS_BOT_ID = 41898282;
  const FAILED = new Set(['failure', 'timed_out']);

  function attemptMarker(run) {
    return `<!-- dynamo-rerun-ci:run:${run.id}:attempt:${run.run_attempt} -->`;
  }

  function trustedReply(comment) {
    return comment.user.id === ACTIONS_BOT_ID && comment.user.type === 'Bot';
  }

  const { issue, comment, action } = context.payload;
  if (context.eventName !== 'issue_comment' || action !== 'created' || !issue.pull_request) return;
  if (!/^\/rerun ci$/i.test(comment.body.trim())) return;

  const repo = context.repo;
  const issueArgs = { ...repo, issue_number: issue.number };
  const eventMarker = `<!-- dynamo-rerun-ci:comment:${comment.id} -->`;
  const comments = await github.paginate(github.rest.issues.listComments, { ...issueArgs, per_page: 100 });
  const replies = comments.filter(trustedReply);
  if (replies.some(reply => reply.body.includes(eventMarker))) return;

  const respond = body => github.rest.issues.createComment({
    ...issueArgs, body: `${eventMarker}\n${body}`,
  });
  // Effective repository permission is authoritative; author_association is not.
  const { data: access } = await github.rest.repos.getCollaboratorPermissionLevel({
    ...repo, username: comment.user.login,
  });
  if (!['write', 'maintain', 'admin'].includes(access.permission)) {
    await respond('The `/rerun ci` command requires repository write access.');
    return;
  }

  const pullArgs = { ...repo, pull_number: issue.number };
  const { data: pull } = await github.rest.pulls.get(pullArgs);
  if (pull.state !== 'open') {
    await respond('This pull request is closed.');
    return;
  }
  const sha = pull.head.sha;
  const branch = `pull-request/${issue.number}`;
  // This protected mirror records copy-pr-bot's approval of this exact commit.
  async function currentApproval(requireCopy) {
    const { data: current } = await github.rest.pulls.get(pullArgs);
    if (current.state !== 'open' || current.head.sha !== sha || current.base.ref !== pull.base.ref) return false;
    if (!requireCopy) return true;
    let mirror;
    try {
      ({ data: mirror } = await github.rest.repos.getBranch({ ...repo, branch }));
    } catch (error) {
      if (error.status === 404) return false;
      throw error;
    }
    return mirror.commit.sha === sha;
  }
  const messages = [];
  const copiedApproved = await currentApproval(true);
  let runs = [];
  if (copiedApproved) {
    runs = await github.paginate(github.rest.actions.listWorkflowRunsForRepo, {
      ...repo, branch, head_sha: sha, event: 'push', per_page: 100,
    });
  } else {
    messages.push('PR and PR-XPU need a copied branch for the current head. Request `/ok to test <sha>` first.');
  }
  const latest = new Map();
  for (const run of runs) {
    if (!WORKFLOWS.has(run.path) || run.head_sha !== sha || run.head_branch !== branch ||
        run.event !== 'push' || run.actor.id !== COPY_PR_BOT_ID) continue;
    if (!latest.has(run.path) || run.id > latest.get(run.path).id) latest.set(run.path, run);
  }

  const preMergeRuns = await github.paginate(github.rest.actions.listWorkflowRuns, {
    ...repo, workflow_id: 'pre-merge.yml', head_sha: sha,
    branch: pull.head.ref, event: 'pull_request', per_page: 100,
  });
  for (const run of preMergeRuns) {
    if (run.path !== '.github/workflows/pre-merge.yml' || run.event !== 'pull_request' ||
        run.head_sha !== sha || run.head_branch !== pull.head.ref) continue;
    // Fork runs can omit PR metadata. Do not guess which PR or target branch they tested.
    if (!run.pull_requests.some(pr => pr.number === issue.number &&
        pr.head.sha === sha && pr.base.ref === pull.base.ref)) {
      messages.push(`[Pre Merge run ${run.id}](${run.html_url}): PR association is unavailable or differs. Retry from Actions if needed.`);
      continue;
    }
    if (!latest.has(run.path) || run.id > latest.get(run.path).id) latest.set(run.path, run);
  }

  const candidates = [];
  for (const run of latest.values()) {
    const link = `[${run.path.split('/').pop()} run ${run.id}](${run.html_url})`;
    // A delayed command cannot select a run or attempt created after that command.
    if (Date.parse(run.updated_at) > Date.parse(comment.created_at)) {
      messages.push(`${link}: changed after this command. Post a new command after completion.`);
    } else if (run.status !== 'completed') {
      messages.push(`${link}: already active.`);
    } else if (Date.now() - Date.parse(run.created_at) >= 30 * 24 * 60 * 60 * 1000 || run.run_attempt >= 50) {
      messages.push(`${link}: outside GitHub's rerun limits. Start a new run through its normal trigger.`);
    } else if (replies.some(reply => reply.body.includes(attemptMarker(run)))) {
      messages.push(`${link}: a retry was already requested for this attempt. Check the run before another retry.`);
    } else {
      const jobs = await github.paginate(github.rest.actions.listJobsForWorkflowRun, {
        ...repo, run_id: run.id, filter: 'latest', per_page: 100,
      });
      if (jobs.some(job => FAILED.has(job.conclusion))) candidates.push({ run, link });
      else messages.push(`${link}: no failed jobs.`);
    }
  }
  if (!latest.size) messages.push('No supported CI runs exist for the current PR commit.');

  // Persist reservations BEFORE retry requests. This also covers delayed API state
  // and workflow reruns after a process failure. User comments cannot forge them.
  const reservations = [];
  const render = () => [
    eventMarker,
    ...reservations,
    `CI retry for \`${sha}\`.`,
    '',
    ...messages.map(line => `- ${line}`),
  ].join('\n');
  const { data: reply } = await github.rest.issues.createComment({
    ...issueArgs, body: render(),
  });
  const update = () => github.rest.issues.updateComment({ ...repo, comment_id: reply.id, body: render() });

  for (const { run, link } of candidates) {
    if (!(await currentApproval(run.event === 'push'))) {
      messages.push('The PR revision or required copied branch changed. Remaining retry requests were stopped.');
      break;
    }
    const { data: fresh } = await github.rest.actions.getWorkflowRun({ ...repo, run_id: run.id });
    if (fresh.status !== 'completed' || fresh.run_attempt !== run.run_attempt) {
      messages.push(`${link}: another retry already started.`);
      continue;
    }
    reservations.push(attemptMarker(run));
    await update();
    try {
      await github.rest.actions.reRunWorkflowFailedJobs({ ...repo, run_id: run.id });
      messages.push(`${link}: requested failed jobs and their dependent jobs.`);
    } catch (error) {
      // Keep the reservation: an interrupted response can hide an accepted POST.
      // Never retry automatically or fall back to a full workflow run.
      messages.push(`${link}: could not confirm the retry. Inspect the run before retrying manually.`);
      await update();
      throw error;
    }
    await update();
  }
  await update();
  core.info(`Processed CI retry command ${comment.id} for PR ${issue.number}.`);
};
