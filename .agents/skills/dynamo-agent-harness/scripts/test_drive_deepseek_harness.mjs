// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { createServer } from 'node:http'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, test } from 'node:test'
import { fileURLToPath } from 'node:url'

const SCRIPT = fileURLToPath(new URL('./drive_deepseek_harness.mjs', import.meta.url))
const temporaryDirectories = []

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) rmSync(directory, { recursive: true, force: true })
})

function temporaryDirectory() {
  const directory = mkdtempSync(join(tmpdir(), 'dsh-dynamo-test-'))
  temporaryDirectories.push(directory)
  return directory
}

function fakeDsh(directory, waitForSignal = false) {
  const path = join(directory, 'fake-dsh.mjs')
  writeFileSync(path, `
const endpoint = process.env.DEEPSEEK_BASE_URL + '/chat/completions'
const response = await fetch(endpoint, {
  method: 'POST',
  headers: {
    authorization: 'Bearer test-secret',
    'content-type': 'application/json',
    'x-deepseek-harness-user-id': 'stable-anonymous-user',
    'x-deepseek-harness-session-id': 'child-session',
    'x-deepseek-harness-parent-session-id': 'parent-session',
    'x-deepseek-harness-compact': '1',
  },
  body: JSON.stringify({ model: 'test-model', messages: [{ role: 'user', content: 'tool result' }], stream: true }),
})
await response.text()
${waitForSignal ? "await new Promise(() => { process.on('SIGINT', () => process.exit(130)); process.on('SIGTERM', () => process.exit(0)) })" : ''}
`)
  return path
}

async function serverHarness(finalStatus = 200) {
  const requests = []
  let notifyRequest
  const firstRequest = new Promise(resolve => { notifyRequest = resolve })
  const server = createServer((request, response) => {
    const chunks = []
    request.on('data', chunk => chunks.push(chunk))
    request.on('end', () => {
      const body = Buffer.concat(chunks).toString('utf8')
      requests.push({ headers: request.headers, body })
      if (request.headers['x-dynamo-session-final'] === 'true') {
        response.writeHead(finalStatus, { 'content-type': 'application/json' })
        response.end(JSON.stringify(finalStatus === 200 ? { ok: true } : { error: 'rejected' }))
      } else {
        notifyRequest()
        response.writeHead(200, { 'content-type': 'text/event-stream' })
        response.end([
          'data: {"choices":[{"delta":{"role":"assistant","content":null,"reasoning_content":""}}]}',
          'data: {"choices":[{"delta":{"content":"hello from mock Dynamo"}}]}',
          'data: {"choices":[{"delta":{"content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":4}}',
          'data: [DONE]',
          '',
        ].join('\n\n'))
      }
    })
  })
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address()
  return {
    baseUrl: `http://127.0.0.1:${address.port}/v1`,
    close: () => new Promise(resolve => server.close(resolve)),
    firstRequest,
    requests,
  }
}

function runWrapper({ baseUrl, capture, dshBin }) {
  const arguments_ = [
    SCRIPT,
    '--base-url', baseUrl,
    '--model', 'test-model',
    '--task', 'run a tool',
    '--capture', capture,
    '--session-final',
  ]
  if (dshBin !== undefined) arguments_.push('--dsh-bin', dshBin)
  const child = spawn(process.execPath, arguments_, {
    env: { ...process.env, DEEPSEEK_API_KEY: 'test-secret' },
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  let stdout = ''
  let stderr = ''
  child.stdout.on('data', chunk => { stdout += chunk })
  child.stderr.on('data', chunk => { stderr += chunk })
  const completed = new Promise((resolve, reject) => {
    child.once('error', reject)
    child.once('close', code => resolve({ code, stderr, stdout }))
  })
  return { child, completed }
}

function evidence(path) {
  return readFileSync(path, 'utf8').trim().split('\n').map(line => JSON.parse(line))
}

test('captures native lineage and sends one canonical final after normal exit', async () => {
  const directory = temporaryDirectory()
  const upstream = await serverHarness()
  try {
    const capture = join(directory, 'capture.jsonl')
    const run = runWrapper({ baseUrl: upstream.baseUrl, capture, dshBin: fakeDsh(directory) })
    const result = await run.completed

    assert.equal(result.code, 0, result.stderr)
    assert.equal(upstream.requests.length, 2)
    assert.equal(upstream.requests[1].headers['x-dynamo-session-id'], 'child-session')
    assert.equal(upstream.requests[1].headers['x-dynamo-session-final'], 'true')
    const records = evidence(capture)
    const request = records.find(record => record.kind === 'request')
    assert.equal(request.headers.authorization, '<redacted>')
    assert.equal(request.headers['x-deepseek-harness-session-id'], 'child-session')
    assert.equal(request.headers['x-deepseek-harness-parent-session-id'], 'parent-session')
    assert.equal(request.headers['x-deepseek-harness-compact'], '1')
    assert.match(request.headers['x-deepseek-harness-user-id'], /^sha256:/)
    assert.deepEqual(records.find(record => record.kind === 'session_final'), {
      timestamp: records.find(record => record.kind === 'session_final').timestamp,
      kind: 'session_final',
      session_id: 'child-session',
      status: 200,
    })
  } finally {
    await upstream.close()
  }
})

test('drains the observed session after SIGINT and exits 130', async () => {
  const directory = temporaryDirectory()
  const upstream = await serverHarness()
  try {
    const capture = join(directory, 'capture.jsonl')
    const run = runWrapper({ baseUrl: upstream.baseUrl, capture, dshBin: fakeDsh(directory, true) })
    await upstream.firstRequest
    run.child.kill('SIGINT')
    const result = await run.completed

    assert.equal(result.code, 130, result.stderr)
    assert.equal(upstream.requests.at(-1).headers['x-dynamo-session-final'], 'true')
    assert.equal(evidence(capture).at(-1).received_signal, 'SIGINT')
  } finally {
    await upstream.close()
  }
})

test('fails closed when ThunderAgent rejects the final request', async () => {
  const directory = temporaryDirectory()
  const upstream = await serverHarness(503)
  try {
    const capture = join(directory, 'capture.jsonl')
    const run = runWrapper({ baseUrl: upstream.baseUrl, capture, dshBin: fakeDsh(directory) })
    const result = await run.completed

    assert.equal(result.code, 1)
    assert.match(result.stderr, /session final .* returned 503/)
    assert.equal(evidence(capture).some(record => record.kind === 'session_final_error'), true)
  } finally {
    await upstream.close()
  }
})

test('runs the published pinned DSH package end to end', { skip: process.env.DSH_PACKAGE_SMOKE !== '1' }, async () => {
  const directory = temporaryDirectory()
  const upstream = await serverHarness()
  try {
    const capture = join(directory, 'capture.jsonl')
    const run = runWrapper({ baseUrl: upstream.baseUrl, capture })
    const result = await run.completed

    assert.equal(result.code, 0, result.stderr)
    assert.match(result.stdout, /hello from mock Dynamo/)
    assert.equal(typeof upstream.requests[0].headers['x-deepseek-harness-session-id'], 'string')
    const final = upstream.requests.find(request => request.headers['x-dynamo-session-final'] === 'true')
    assert.equal(final.headers['x-dynamo-session-id'], upstream.requests[0].headers['x-deepseek-harness-session-id'])
  } finally {
    await upstream.close()
  }
})

test('runs a built patched DSH source tree end to end', { skip: process.env.DSH_PATCHED_BIN === undefined }, async () => {
  const directory = temporaryDirectory()
  const upstream = await serverHarness()
  try {
    const capture = join(directory, 'capture.jsonl')
    const run = runWrapper({ baseUrl: upstream.baseUrl, capture, dshBin: process.env.DSH_PATCHED_BIN })
    const result = await run.completed

    assert.equal(result.code, 0, result.stderr)
    assert.match(result.stdout, /hello from mock Dynamo/)
    assert.equal(typeof upstream.requests[0].headers['x-deepseek-harness-session-id'], 'string')
    const final = upstream.requests.find(request => request.headers['x-dynamo-session-final'] === 'true')
    assert.equal(final.headers['x-dynamo-session-id'], upstream.requests[0].headers['x-deepseek-harness-session-id'])
  } finally {
    await upstream.close()
  }
})
