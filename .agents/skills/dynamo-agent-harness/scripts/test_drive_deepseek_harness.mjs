// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { createServer } from 'node:http'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { afterEach, test } from 'node:test'
import { fileURLToPath } from 'node:url'

const SCRIPT = fileURLToPath(new URL('./drive_deepseek_harness.mjs', import.meta.url))
const { normalizeBaseUrl } = await import(new URL('./drive_deepseek_harness.mjs', import.meta.url))
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
import { join } from 'node:path'
import { writeFileSync } from 'node:fs'

writeFileSync(join(process.env.DSH_HOME, 'child-env.json'), JSON.stringify({
  ci: process.env.CI ?? null,
  corepackDownloadPrompt: process.env.COREPACK_ENABLE_DOWNLOAD_PROMPT ?? null,
  corepackHome: process.env.COREPACK_HOME ?? null,
  credential: process.env.DEEPSEEK_API_KEY,
  dynamoCredential: process.env.DYNAMO_API_KEY ?? null,
  home: process.env.HOME,
  leakedCredential: process.env.LEAK_TEST_SECRET ?? null,
}))
const endpoint = process.env.DEEPSEEK_BASE_URL + '/chat/completions'
const response = await fetch(endpoint, {
  method: 'POST',
  headers: {
    authorization: 'Bearer ' + process.env.DEEPSEEK_API_KEY,
    'content-type': 'application/json',
    'x-deepseek-harness-user-id': 'stable-anonymous-user',
    'x-deepseek-harness-session-id': 'dsh-session',
    'x-deepseek-harness-compact': '1',
  },
  body: JSON.stringify({ model: 'test-model', messages: [{ role: 'user', content: 'tool result' }], stream: true }),
})
await response.text()
${waitForSignal ? "await new Promise(() => { process.on('SIGINT', () => process.exit(130)); process.on('SIGTERM', () => process.exit(0)) })" : ''}
`)
  return path
}

function fakeDshWithoutRequest(directory) {
  const path = join(directory, 'fake-dsh-no-request.mjs')
  writeFileSync(path, "process.stdout.write('exited without a model request\\n')\n")
  return path
}

function localDshPackage(directory) {
  const packageDirectory = join(directory, 'fake-dsh-package')
  mkdirSync(packageDirectory)
  writeFileSync(join(packageDirectory, 'package.json'), `${JSON.stringify({
    name: 'fake-dsh-process-tree',
    version: '1.0.0',
    type: 'module',
    bin: { 'fake-dsh-process-tree': 'bin.mjs' },
  }, null, 2)}\n`)
  const bin = join(packageDirectory, 'bin.mjs')
  writeFileSync(bin, `#!/usr/bin/env node
import { spawn } from 'node:child_process'
import { join } from 'node:path'
import { writeFileSync } from 'node:fs'

const pidPath = join(process.env.DSH_HOME, 'descendant-pid')
writeFileSync(pidPath, String(process.pid))
const stubborn = spawn('/bin/sh', ['-c', "trap '' TERM; while :; do sleep 1; done"], {
  detached: true,
  stdio: 'ignore',
})
writeFileSync(join(process.env.DSH_HOME, 'stubborn-pid'), String(stubborn.pid))
stubborn.unref()
const interrupted = new Promise(() => {
  for (const signal of ['SIGINT', 'SIGTERM']) {
    process.on(signal, () => {
      writeFileSync(join(process.env.DSH_HOME, 'descendant-signal'), signal)
      process.exit(signal === 'SIGINT' ? 130 : 143)
    })
  }
})
const endpoint = process.env.DEEPSEEK_BASE_URL + '/chat/completions'
const response = await fetch(endpoint, {
  method: 'POST',
  headers: {
    authorization: 'Bearer ' + process.env.DEEPSEEK_API_KEY,
    'content-type': 'application/json',
    'x-deepseek-harness-session-id': 'process-tree-session',
  },
  body: JSON.stringify({ model: 'test-model', messages: [{ role: 'user', content: 'wait for signal' }], stream: true }),
})
await response.text()
await interrupted
`)
  chmodSync(bin, 0o755)
  return `file:${resolve(packageDirectory)}`
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

function runWrapper({ baseUrl, capture, dshBin, dshHome, environment = {}, extraArguments = [], sessionFinal = true }) {
  const arguments_ = [
    SCRIPT,
    '--base-url', baseUrl,
    '--model', 'test-model',
    '--task', 'run a tool',
    '--capture', capture,
  ]
  if (sessionFinal) arguments_.push('--session-final')
  if (dshBin !== undefined) arguments_.push('--dsh-bin', dshBin)
  if (dshHome !== undefined) arguments_.push('--dsh-home', dshHome)
  arguments_.push(...extraArguments)
  const child = spawn(process.execPath, arguments_, {
    env: {
      ...process.env,
      COREPACK_HOME: '/sensitive/shared-corepack-cache',
      DEEPSEEK_API_KEY: 'ambient-deepseek-secret',
      DYNAMO_API_KEY: 'test-dynamo-secret',
      LEAK_TEST_SECRET: 'must-not-reach-dsh',
      ...environment,
    },
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

async function waitFor(predicate, description) {
  const deadline = Date.now() + 5_000
  while (!predicate()) {
    if (Date.now() >= deadline) throw new Error(`timed out waiting for ${description}`)
    await new Promise(resolveWait => setTimeout(resolveWait, 20))
  }
}

function processIsAlive(pid) {
  try {
    process.kill(pid, 0)
    return true
  } catch (error) {
    if (error instanceof Error && 'code' in error && error.code === 'ESRCH') return false
    throw error
  }
}

test('normalizes root and versioned Dynamo endpoints without a protocol-relative path', () => {
  assert.equal(normalizeBaseUrl('http://127.0.0.1:8000').toString(), 'http://127.0.0.1:8000/v1')
  assert.equal(normalizeBaseUrl('http://127.0.0.1:8000/').toString(), 'http://127.0.0.1:8000/v1')
  assert.equal(normalizeBaseUrl('http://127.0.0.1:8000/v1/').toString(), 'http://127.0.0.1:8000/v1')
})

test('captures native session metadata and sends one canonical final after normal exit', async () => {
  const directory = temporaryDirectory()
  const upstream = await serverHarness()
  try {
    const capture = join(directory, 'capture.jsonl')
    const dshHome = join(directory, 'dsh-home')
    const run = runWrapper({ baseUrl: upstream.baseUrl, capture, dshBin: fakeDsh(directory), dshHome })
    const result = await run.completed

    assert.equal(result.code, 0, result.stderr)
    assert.equal(upstream.requests.length, 2)
    assert.equal(upstream.requests[0].headers.authorization, 'Bearer test-dynamo-secret')
    assert.equal(upstream.requests[0].headers['x-dynamo-session-id'], undefined)
    assert.equal(upstream.requests[1].headers['x-dynamo-session-id'], 'dsh-session')
    assert.equal(upstream.requests[1].headers['x-dynamo-session-final'], 'true')
    const records = evidence(capture)
    const request = records.find(record => record.kind === 'request')
    assert.equal(request.headers.authorization, '<redacted>')
    assert.equal(request.headers['x-deepseek-harness-session-id'], 'dsh-session')
    assert.equal(request.headers['x-deepseek-harness-compact'], '1')
    assert.match(request.headers['x-deepseek-harness-user-id'], /^sha256:/)
    assert.deepEqual(records.find(record => record.kind === 'session_final'), {
      timestamp: records.find(record => record.kind === 'session_final').timestamp,
      kind: 'session_final',
      session_id: 'dsh-session',
      status: 200,
    })
    assert.deepEqual(JSON.parse(readFileSync(join(dshHome, 'child-env.json'), 'utf8')), {
      ci: '1',
      corepackDownloadPrompt: '0',
      corepackHome: null,
      credential: 'test-dynamo-secret',
      dynamoCredential: null,
      home: dshHome,
      leakedCredential: null,
    })
  } finally {
    await upstream.close()
  }
})

test('adds canonical identity only when the Dynamo 1.3 compatibility bridge is selected', async () => {
  const directory = temporaryDirectory()
  const upstream = await serverHarness()
  try {
    const capture = join(directory, 'capture.jsonl')
    const run = runWrapper({
      baseUrl: upstream.baseUrl,
      capture,
      dshBin: fakeDsh(directory),
      dshHome: join(directory, 'dsh-home'),
      extraArguments: ['--canonicalize-dynamo-headers'],
      sessionFinal: false,
    })
    const result = await run.completed

    assert.equal(result.code, 0, result.stderr)
    assert.equal(upstream.requests[0].headers['x-deepseek-harness-session-id'], 'dsh-session')
    assert.equal(upstream.requests[0].headers['x-dynamo-session-id'], 'dsh-session')
    assert.equal(evidence(capture).find(record => record.kind === 'run_start').canonicalize_dynamo_headers, true)
  } finally {
    await upstream.close()
  }
})

test('uses only the explicitly selected credential environment variable', async () => {
  const directory = temporaryDirectory()
  const upstream = await serverHarness()
  try {
    const capture = join(directory, 'capture.jsonl')
    const run = runWrapper({
      baseUrl: upstream.baseUrl,
      capture,
      dshBin: fakeDsh(directory),
      environment: { SELECTED_DYNAMO_KEY: 'explicit-selected-secret' },
      extraArguments: ['--api-key-env', 'SELECTED_DYNAMO_KEY'],
    })
    const result = await run.completed

    assert.equal(result.code, 0, result.stderr)
    assert.equal(upstream.requests[0].headers.authorization, 'Bearer explicit-selected-secret')
    assert.equal(evidence(capture).find(record => record.kind === 'run_start').api_key_env, 'SELECTED_DYNAMO_KEY')
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

test('force-kills an uncooperative detached descendant in the real Corepack and pnpm tree and exits 143', async () => {
  const directory = temporaryDirectory()
  const upstream = await serverHarness()
  try {
    const capture = join(directory, 'capture.jsonl')
    const dshHome = join(directory, 'dsh-home')
    const run = runWrapper({
      baseUrl: upstream.baseUrl,
      capture,
      dshHome,
      extraArguments: ['--dsh-package', localDshPackage(directory)],
    })
    await upstream.firstRequest
    const descendantPid = Number(readFileSync(join(dshHome, 'descendant-pid'), 'utf8'))
    const stubbornPid = Number(readFileSync(join(dshHome, 'stubborn-pid'), 'utf8'))
    assert.equal(processIsAlive(descendantPid), true)
    assert.equal(processIsAlive(stubbornPid), true)

    run.child.kill('SIGTERM')
    const result = await run.completed
    const records = evidence(capture)
    const initialSignal = records.find(record => record.kind === 'process_tree_signal')
    const forceKill = records.find(record => record.kind === 'process_tree_force_kill')
    assert.ok(initialSignal?.tracked_processes >= 3, JSON.stringify(records, null, 2))
    assert.ok(forceKill?.tracked_processes >= 3, JSON.stringify(records, null, 2))
    await waitFor(() => !processIsAlive(descendantPid), 'the DSH descendant to exit')
    await waitFor(() => !processIsAlive(stubbornPid), 'the detached descendant to exit')

    assert.equal(result.code, 143, result.stderr)
    assert.equal(upstream.requests.at(-1).headers['x-dynamo-session-final'], 'true')
    assert.equal(records.some(record => record.kind === 'process_tree_force_kill'), true)
    assert.equal(records.at(-1).received_signal, 'SIGTERM')
  } finally {
    await upstream.close()
  }
})

test('fails closed when lifecycle mode observes zero sessions', async () => {
  const directory = temporaryDirectory()
  const upstream = await serverHarness()
  try {
    const capture = join(directory, 'capture.jsonl')
    const run = runWrapper({ baseUrl: upstream.baseUrl, capture, dshBin: fakeDshWithoutRequest(directory) })
    const result = await run.completed

    assert.equal(result.code, 1)
    assert.match(result.stderr, /zero DSH sessions reached Dynamo/)
    assert.equal(upstream.requests.length, 0)
    assert.equal(evidence(capture).some(record => record.kind === 'session_final_error'), true)
  } finally {
    await upstream.close()
  }
})

test('refuses to replace capture evidence without explicit overwrite', async () => {
  const directory = temporaryDirectory()
  const upstream = await serverHarness()
  try {
    const capture = join(directory, 'capture.jsonl')
    writeFileSync(capture, 'preserve-this-evidence\n')
    const refused = runWrapper({ baseUrl: upstream.baseUrl, capture, dshBin: fakeDsh(directory) })
    const refusedResult = await refused.completed

    assert.equal(refusedResult.code, 1)
    assert.match(refusedResult.stderr, /EEXIST/)
    assert.equal(readFileSync(capture, 'utf8'), 'preserve-this-evidence\n')
    assert.equal(upstream.requests.length, 0)

    const overwrite = runWrapper({
      baseUrl: upstream.baseUrl,
      capture,
      dshBin: fakeDsh(directory),
      extraArguments: ['--overwrite-capture'],
    })
    const overwriteResult = await overwrite.completed
    assert.equal(overwriteResult.code, 0, overwriteResult.stderr)
    assert.equal(evidence(capture).some(record => record.kind === 'run_start'), true)
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
