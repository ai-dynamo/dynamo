// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

module.exports = async function rerunFailedCI({ github, context, core }) {
  const COPY_PR_BOT_ID = 138252374;
  const ACTIONS_BOT_ID = 41898282;
  const FAILED = new Set(['failure', 'timed_out']);
  const MAX_RUN_AGE_MS = 7 * 24 * 60 * 60 * 1000;

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

  try {
    await github.rest.reactions.createForIssueComment({
      ...repo, comment_id: comment.id, content: '+1',
    });
  } catch {
    core.warning(`Could not acknowledge CI retry command ${comment.id}. Continuing without a reaction.`);
  }

  const pullArgs = { ...repo, pull_number: issue.number };
  const { data: pull } = await github.rest.pulls.get(pullArgs);
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
  const runs = await github.paginate(github.rest.actions.listWorkflowRunsForRepo, {
    ...repo, head_sha: sha, branch: pull.head.ref, event: 'pull_request', per_page: 100,
  });
  runs.push(...await github.paginate(github.rest.actions.listWorkflowRunsForRepo, {
    ...repo, head_sha: sha, branch, event: 'push', per_page: 100,
  }));
  const latest = new Map();
  for (const run of runs) {
    const copiedRun = run.event === 'push' && run.head_branch === branch &&
      run.actor.id === COPY_PR_BOT_ID;
    const pullRun = run.event === 'pull_request' && run.head_branch === pull.head.ref &&
      run.pull_requests.some(pr => pr.number === issue.number &&
        pr.head.sha === sha && pr.base.ref === pull.base.ref);
    if (!copiedRun && !pullRun) continue;
    if (!latest.has(run.path) || run.id > latest.get(run.path).id) latest.set(run.path, run);
  }

  const candidates = [];
  let failedRequests = 0;
  const existingRuns = [...latest.values()];
  for (const run of existingRuns) {
    const link = `[${run.path.split('/').pop()} run ${run.id}](${run.html_url})`;
    // A delayed command cannot select a run or attempt created after that command.
    if (Date.parse(run.updated_at) > Date.parse(comment.created_at)) {
      messages.push(`${link}: changed after this command. Post a new command after completion.`);
    } else if (run.status !== 'completed') {
      messages.push(`${link}: already active.`);
    } else if (Date.now() - Date.parse(run.created_at) >= MAX_RUN_AGE_MS) {
      messages.push(`${link}: outside the seven-day retry window. Start a new run through its normal trigger.`);
    } else if (replies.some(reply => reply.body.includes(attemptMarker(run)))) {
      messages.push(`${link}: a retry was already requested for this attempt. Check the run before another retry.`);
    } else {
      try {
        const jobs = await github.paginate(github.rest.actions.listJobsForWorkflowRun, {
          ...repo, run_id: run.id, filter: 'latest', per_page: 100,
        });
        if (jobs.some(job => FAILED.has(job.conclusion))) {
          candidates.push({ run, link, messageIndex: messages.length });
          messages.push(`${link}: not requested yet.`);
        } else messages.push(`${link}: no failed jobs.`);
      } catch (error) {
        core.warning(`Could not list jobs for workflow run ${run.id} (HTTP ${error.status ?? 'unknown'}).`);
        messages.push(`${link}: could not inspect jobs. No retry was requested. Post a new command to try again.`);
        failedRequests++;
      }
    }
  }
  if (!existingRuns.length) messages.push('No matching CI runs were found for the current PR commit.');

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
  const update = async () => {
    try {
      await github.rest.issues.updateComment({ ...repo, comment_id: reply.id, body: render() });
      return true;
    } catch (error) {
      core.warning(`Could not update the PR reply (HTTP ${error.status ?? 'unknown'}).`);
      return false;
    }
  };

  for (const [index, { run, link, messageIndex }] of candidates.entries()) {
    if (!(await currentApproval(run.event === 'push'))) {
      for (const pending of candidates.slice(index)) {
        messages[pending.messageIndex] = `${pending.link}: not requested because the PR revision or required copied branch changed.`;
      }
      break;
    }
    let fresh;
    try {
      ({ data: fresh } = await github.rest.actions.getWorkflowRun({ ...repo, run_id: run.id }));
    } catch (error) {
      core.warning(`Could not inspect workflow run ${run.id} (HTTP ${error.status ?? 'unknown'}).`);
      messages[messageIndex] = `${link}: could not inspect the current attempt. No retry was requested. Post a new command to try again.`;
      failedRequests++;
      await update();
      continue;
    }
    if (fresh.status !== 'completed' || fresh.run_attempt !== run.run_attempt) {
      messages[messageIndex] = `${link}: another retry already started.`;
      continue;
    }
    reservations.push(attemptMarker(run));
    if (!(await update())) {
      reservations.pop();
      messages[messageIndex] = `${link}: could not record the retry reservation. No retry was requested. Post a new command to try again.`;
      failedRequests++;
      continue;
    }
    try {
      await github.rest.actions.reRunWorkflowFailedJobs({ ...repo, run_id: run.id });
      messages[messageIndex] = `${link}: requested failed jobs and their dependent jobs.`;
    } catch (error) {
      core.warning(`Could not request a retry for workflow run ${run.id} (HTTP ${error.status ?? 'unknown'}).`);
      // An interrupted response can hide an accepted POST. Keep its reservation.
      // Never retry automatically or fall back to a full workflow run.
      messages[messageIndex] = `${link}: could not confirm the retry. Reservation retained. Inspect the run before retrying manually in Actions.`;
      failedRequests++;
    }
    await update();
  }
  await update();
  if (failedRequests) core.setFailed(`Could not complete ${failedRequests} workflow retry request(s). See the PR reply.`);
  core.info(`Processed CI retry command ${comment.id} for PR ${issue.number}.`);
};
