// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
'use strict';
const fs = require('node:fs');
const YAML = require('yaml');
const {eligibleAuthors} = require('./route.cjs');
async function main() {
  const event = JSON.parse(fs.readFileSync(process.env.GITHUB_EVENT_PATH, 'utf8'));
  if (process.env.GITHUB_REPOSITORY !== 'ai-dynamo/dynamo' || process.env.GITHUB_EVENT_NAME !== 'pull_request_target') {
    throw new Error('Labeling requires a trusted metadata event');
  }
  const allowed = eligibleAuthors(YAML.parse(fs.readFileSync('contributors.yaml', 'utf8')));
  const label = 'trusted-contributor';
  const root = `https://api.github.com/repos/ai-dynamo/dynamo/issues/${event.number}/labels`;
  const approved = allowed.has(event.pull_request.user.login.toLowerCase());
  const response = await fetch(approved ? root : `${root}/${label}`, {
    method: approved ? 'POST' : 'DELETE', headers: {
      Authorization: `Bearer ${process.env.GH_TOKEN}`, Accept: 'application/vnd.github+json',
      'Content-Type': 'application/json'}, body: approved ? JSON.stringify({labels: [label]}) : undefined,
    signal: AbortSignal.timeout(20000)});
  if (!response.ok && !(response.status === 404 && !approved)) throw new Error(`Label update failed: ${response.status}`);
}
main().catch(error => { console.error(error.message); process.exitCode = 1; });
