# Dynamo Full Docs Refactor Backup Record

This file records the recoverable history immediately before the branch was squashed on July 24, 2026.
It is intentionally tracked in the squashed branch so the original commits and backup refs remain easy to find.

## Recovery Points

| Purpose | Branch | Commit |
| --- | --- | --- |
| Before merging PR #8 (`dagil/reference-components-v1.3.0`) | `origin/backup/dynamo-full-docs-refactor-pre-pr8-20260724` | `92df7d3f89738712bd2a80dea687749a892a296e` |
| After merging PR #8 and immediately before squashing | `origin/backup/dynamo-full-docs-refactor-pre-squash-20260724` | `785567915663fbb753a16cdd3faea243fcdbcf73` |

Both backup branches were pushed to `Jont828/dynamo` on July 24, 2026.

## Pre-Squash State

- Branch: `dynamo-full-docs-refactor`
- Head commit: `785567915663fbb753a16cdd3faea243fcdbcf73`
- Head tree: `370835a43993fed3f62800245bb117e17e29454b`
- Upstream comparison ref at inventory time: `upstream/main` at `83173da5bbdd033044ec89b524ddc2c3b72d36fa`
- Squash base / merge base: `b1c95e41605cf316ab85883c9f49b1616f13156f`
- Commits reachable from the branch after the squash base: **80**
- First-parent commits after the squash base: **56**

Squash base:

- Date: `2026-06-22T11:06:09-07:00`
- Author: `Krishnan Prashanth <140860868+KrishnanPrash@users.noreply.github.com>`
- Subject: `fix(llm): fall back to tokenizer.json for eos/bos token ids (#10811)`

Pre-squash head:

- Date: `2026-07-24T18:05:54-04:00`
- Author: `Jont828 <jt572@cornell.edu>`
- Subject: `docs(reference): merge v1.3.0 reference components`

## Restore Commands

Restore the complete post-PR, pre-squash history:

```bash
git fetch origin backup/dynamo-full-docs-refactor-pre-squash-20260724
git switch -C dynamo-full-docs-refactor-restored origin/backup/dynamo-full-docs-refactor-pre-squash-20260724
```

Restore the branch from before PR #8 was merged:

```bash
git fetch origin backup/dynamo-full-docs-refactor-pre-pr8-20260724
git switch -C dynamo-full-docs-refactor-before-pr8 origin/backup/dynamo-full-docs-refactor-pre-pr8-20260724
```

Verify that a restored checkout exactly matches the recorded pre-squash tree:

```bash
test "$(git rev-parse HEAD^{tree})" = "370835a43993fed3f62800245bb117e17e29454b"
```

## Contributors Preserved on the Squashed Commit

The squashed commit carries `Co-authored-by` trailers for all other human contributors whose work is represented in this branch:

| Contributor | Email | Basis |
| --- | --- | --- |
| alimaazamat | `alima.azamat2003@gmail.com` | Co-author trailer on the branch's initial docs reorganization commit |
| akshatha-k | `akshutk@gmail.com` | Five authored commits in the branch-only history |
| Dan Gil | `dagil@nvidia.com` | Upstream Reference-tab work merged from dagil/reference-components-v1.3.0 |
| Ben Hamm | `ben.hamm@gmail.com` | Recipe and Fern migration commits merged into the branch |
| Harry Kim | `harryk@nvidia.com` | Community events work merged into the branch |

## Commit Inventory

The following list contains every commit reachable from the pre-squash branch after the recorded merge base, including commits brought in through merged branches.
Full commit messages are retained below each entry.

### 1. `724168c014d5` — docs: reorganize documentation site with tabbed nav, landing pages, and reference sections

- Commit: `724168c014d5c33d492a0adc75e6e9984401472d`
- Parent(s): `b1c95e41605cf316ab85883c9f49b1616f13156f`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-06T19:50:06-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-06T20:17:31-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>docs: reorganize documentation site with tabbed nav, landing pages, and reference sections

Overhaul the Fern docs site structure and expand Kubernetes and
observability reference material.

- Restructure nav into tabs with new sidebar sections and Font Awesome icons
- Add welcome/landing page with gradient and GitHub buttons
- Refactor Kubernetes model deployment, DGD/DGDR guides, and add references
- Add multinode installation and installation guide updates
- Add observability reference pages (env vars, metric labels, metrics
  catalog, operator metrics, local resource monitor)
- Convert compatibility and release-artifacts pages to MDX
- Add backend templates (sglang, trtllm, vllm) and tools docs (aic, aiperf)

Co-authored-by: alimaazamat &lt;alima.azamat2003@gmail.com&gt;
Co-authored-by: akshatha-k &lt;akshutk@gmail.com&gt;
Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 2. `085a77edd06c` — Update badges in K8s doc enums

- Commit: `085a77edd06cb859bfb758322cc8358bd42760e6`
- Parent(s): `724168c014d5c33d492a0adc75e6e9984401472d`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-06T21:34:04-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-06T21:34:04-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>Update badges in K8s doc enums

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 3. `698b4cb46d01` — Work on rewriting profiler/planner guides

- Commit: `698b4cb46d0173911fd3c5b11410be4c5288f0b8`
- Parent(s): `085a77edd06cb859bfb758322cc8358bd42760e6`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-07T18:11:43-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-07T18:11:43-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>Work on rewriting profiler/planner guides

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 4. `d319d989d5b3` — Refactor observability, add use cases tab, and work on dgdr guide section

- Commit: `d319d989d5b3b19926fa0673ab0bd1692ca672c6`
- Parent(s): `698b4cb46d0173911fd3c5b11410be4c5288f0b8`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-07T22:17:59-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-07T22:17:59-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>Refactor observability, add use cases tab, and work on dgdr guide section

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 5. `08db9ac2a682` — Add planner reference guide

- Commit: `08db9ac2a682a6a79c813bc7824412bcba926621`
- Parent(s): `d319d989d5b3b19926fa0673ab0bd1692ca672c6`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-08T13:12:46-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-08T13:16:23-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>Add planner reference guide

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 6. `15c8474e202e` — add tab

- Commit: `15c8474e202e7018336010673e59415a03642abe`
- Parent(s): `08db9ac2a682a6a79c813bc7824412bcba926621`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-08T13:28:03-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-08T13:28:03-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>add tab

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 7. `0aa1dccb4341` — Work on refactoring fault tolerance, tool call, and inference sim/dynosim

- Commit: `0aa1dccb4341e575d367bf5e74c32d32c3b5a913`
- Parent(s): `15c8474e202e7018336010673e59415a03642abe`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-08T17:34:34-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-08T17:34:34-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>Work on refactoring fault tolerance, tool call, and inference sim/dynosim

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 8. `565b20099edb` — docs: add TerminalDemo hero component to welcome page

- Commit: `565b20099edbb236e4ccc1c5571d0bdb8190fa1a`
- Parent(s): `0aa1dccb4341e575d367bf5e74c32d32c3b5a913`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-10T21:16:16-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-10T21:16:16-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>docs: add TerminalDemo hero component to welcome page

Add a CDN-loaded asciinema-player component (fern/components/TerminalDemo.tsx)
that renders a looping, autoplaying terminal recording in a macOS-style window
frame, and wire it into the welcome page as a hero demo.

- Move welcome.mdx into fern/ so the @/components import resolves in local dev
  (Option 1: page under the fern root, works with ./fern/watch.sh).
- Ship the recording at docs/assets/hero-demo.cast (GitHub Dark palette embedded
  in the header); the player fetches it via a hosted asciinema URL.
- Playback-controls toggle in the window bar: reveals the control bar with a
  slide-out that expands the window, without re-creating the player.
- Reserve terminal height before load to avoid a 0-height flash; release it once
  the player mounts so the controls reveal can grow the box.

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 9. `e9dd707d1852` — docs: restructure Tool Calling & Reasoning section

- Commit: `e9dd707d1852598b1fc9d794294ca53fc6100a69`
- Parent(s): `565b20099edbb236e4ccc1c5571d0bdb8190fa1a`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-10T21:18:56-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-10T21:18:56-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>docs: restructure Tool Calling &amp; Reasoning section

Rework the tool-calling and reasoning docs around a clear two-layer model:
the chat processor always runs first (default `dynamo`), then tool-call and
reasoning parsing are independent, optional features.

- Add a dedicated Chat Processors page covering `--dyn-chat-processor` and
  engine fallback, with vLLM/SGLang examples
- Rewrite the Introduction to a two-step Steps flow (pick a chat processor,
  then choose parsers)
- Make the Tool Call and Reasoning pages instructional: &quot;to enable X, add
  flag Y to the DGD as follows&quot; with an abbreviated DGD spec skeleton, then
  the supported-values table
- Keep the zh-CN mirror in sync and update inbound links/nav

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 10. `9dccdab4b5be` — docs: rework Fault Tolerance overview as plain-English Introduction

- Commit: `9dccdab4b5bea5dc7f03536942331ef2741da5b2`
- Parent(s): `e9dd707d1852598b1fc9d794294ca53fc6100a69`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-10T22:08:30-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-10T22:08:30-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>docs: rework Fault Tolerance overview as plain-English Introduction

Reframe the section landing page around user-configurable behaviors
(migration, rejection, graceful shutdown) vs. built-in runtime behaviors,
in plain language with a bullet list. Drop the redundant config quick-
reference (now covered in the Reference tab) and the automatic failure-
scenario tables, and rename the nav entry from Overview to Introduction.

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 11. `1c3bcafb5b2f` — docs: rework Chat Processors page into a three-tab config view

- Commit: `1c3bcafb5b2ffcc44f0051608fa0b70a777809ae`
- Parent(s): `9dccdab4b5bea5dc7f03536942331ef2741da5b2`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-10T22:28:16-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-10T22:28:16-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>docs: rework Chat Processors page into a three-tab config view

Present the three chat-processor options (dynamo default, vLLM, SGLang) as a
single Tabs view with abbreviated DGD specs, dropping the separate flag and
pairing-rule sections.

- Lead with a concise concept + three-option summary; recommend the dynamo
  default whenever a parser exists
- Each engine tab notes that vLLM/SGLang parsing is an engine fallback and
  warns that the parser flags belong on the Frontend, not the worker
- Abbreviate DGD snippets with &quot;# ...&quot; comments for irrelevant fields
- Remove the standalone Engine fallback section and retarget inbound
  #engine-fallback anchor links to the page

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 12. `273779c50b58` — docs(welcome): add hero terminal casts + recording tooling

- Commit: `273779c50b58de51cf5cd99745dd1964c3a3e55c`
- Parent(s): `1c3bcafb5b2ffcc44f0051608fa0b70a777809ae`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-10T22:39:31-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-10T22:39:31-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>docs(welcome): add hero terminal casts + recording tooling

Commit the GitHub Dark Default-themed hero recordings and the tooling to
re-record them, alongside the TerminalDemo/welcome tweaks (local-asset src,
thinner title bar, #0d1117 bg, 2.16 aspect reserve).

- docs/assets/hero-demo-25.cast: the served 120x25 hero (welcome.mdx src)
- fern/hero-demo/: 120x25/28/32 casts + hero-demo.sh, record-hero.sh,
  apply-hero-theme.py (embeds the palette), and hero-demo-plan.md

Palette rides in each cast&#x27;s term.theme header (bg #0d1117, red #ff7b72,
green #3fb950, classic yellow #e3b341, blue #58a6ff), so the player uses it
over the component default.

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 13. `a9e61f2acd4a` — Working on diffusion and dynosim refactor

- Commit: `a9e61f2acd4a83fd69a2bd7a885cdaa62e882b40`
- Parent(s): `273779c50b58de51cf5cd99745dd1964c3a3e55c`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-10T22:45:29-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-10T22:45:29-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>Working on diffusion and dynosim refactor

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 14. `f5ad84b9e5b1` — Update palatte and cast files

- Commit: `f5ad84b9e5b1a8c5757cffe6760b2321adf93755`
- Parent(s): `a9e61f2acd4a83fd69a2bd7a885cdaa62e882b40`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-10T23:18:46-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-10T23:18:46-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>Update palatte and cast files

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 15. `78c06cb7d7f3` — Edits to diffusion, toolcall, and add more reference pages. Split use cases to include key features section

- Commit: `78c06cb7d7f3539fdb49bf6f049a95fedc5bb907`
- Parent(s): `f5ad84b9e5b1a8c5757cffe6760b2321adf93755`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-13T20:52:55-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-13T20:52:55-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>Edits to diffusion, toolcall, and add more reference pages. Split use cases to include key features section

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 16. `18ca2b95faee` — docs(tool-calling): fold probe snapshot into Tool Call Parsing page

- Commit: `18ca2b95faeef4f841d7d52287f7db98495d4fdc`
- Parent(s): `78c06cb7d7f3539fdb49bf6f049a95fedc5bb907`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-14T19:24:47-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-14T19:24:47-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>docs(tool-calling): fold probe snapshot into Tool Call Parsing page

Convert tool-calling/README.md to .mdx so it can render Fern components,
retargeting all inbound README.md links (index.yml nav, chat-processors,
introduction, troubleshooting, structural-tag, zh-CN mirror).

Move the hidden Dynamo 1.2 tool-calling probe snapshot to the bottom of the
Tool Call Parsing page as a per-model &lt;Tabs&gt; group, and delete the standalone
hidden reference page plus its nav entry. The probe measures tool-calling
workflows; reasoning appears only where a model emits both in one native
format or where the parser misroutes assistant text into the reasoning channel.

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 17. `b6f6edb824fa` — docs(diffusion): regroup diffusion docs by use case with per-backend subpages

- Commit: `b6f6edb824fa05bc280ba9f47927ee9b39cf222d`
- Parent(s): `18ca2b95faeef4f841d7d52287f7db98495d4fdc`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-14T20:25:34-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-14T20:25:34-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>docs(diffusion): regroup diffusion docs by use case with per-backend subpages

Reorganize the diffusion documentation modality-first (text-to-image,
text-to-video, image-to-video, text-to-audio, text-to-text), each with a
per-backend subpage (vLLM-Omni, SGLang, TensorRT-LLM, FastVideo). The
Overview page now covers every backend and folds their install steps and
requirements into a per-backend Tab group, plus a full support matrix.

Move the vLLM-Omni disaggregated multi-stage architecture into the
Knowledge Base design docs. Remove the old omni/ section and the
per-backend sglang-diffusion / trtllm-diffusion pages, repointing all
inbound links. Also reorder the Features tab so Use Cases sits above
Key Features.

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 18. `f381976189ee` — docs: add community meeting, CNCF, and Slack info to Community page

- Commit: `f381976189eebc1a5ff66fd7b274546fe1912573`
- Parent(s): `b6f6edb824fa05bc280ba9f47927ee9b39cf222d`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-14T20:30:29-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-14T20:30:29-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>docs: add community meeting, CNCF, and Slack info to Community page

Add the weekly community meeting time, Google Meet link, and notes doc to
the Community page, plus a downloadable .ics calendar invite. Link the CNCF
Slack invite and CNCF join page, and fix the welcome page Slack card to use
the working invite URL.

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 19. `122151f72f1f` — docs: restructure nav tabs and add self-contained CLI Guide

- Commit: `122151f72f1f934a2b092fef462b1a6bf1d2299c`
- Parent(s): `f381976189eebc1a5ff66fd7b274546fe1912573`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-14T21:11:08-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-14T21:11:08-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>docs: restructure nav tabs and add self-contained CLI Guide

Rename tabs (User Guide -&gt; Kubernetes Guide, Use Cases -&gt; Features) and
update their icons. Add a CLI Guide tab with its own `cli` URL slug and
Getting Started / Installation / Model Deployment / Operations sections.

Give the CLI Guide its own tailored pages under docs/cli/ (introduction,
overview, building-from-source, observability, sglang-hicache) so no page
is reused across tabs; strip Kubernetes-specific content from the local
observability and health-check pages.

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 20. `899dfa9c78ef` — Minor edits to diffusion/toolcall

- Commit: `899dfa9c78ef2000e5e470ea68803d79addb42af`
- Parent(s): `122151f72f1f934a2b092fef462b1a6bf1d2299c`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-14T21:11:38-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-14T21:11:38-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>Minor edits to diffusion/toolcall

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 21. `31cf60a77646` — Dev scripts, can remove later

- Commit: `31cf60a77646f335290903e534b938cac8eecd88`
- Parent(s): `899dfa9c78ef2000e5e470ea68803d79addb42af`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-17T15:28:17-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-17T15:28:17-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>Dev scripts, can remove later

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 22. `7674ddd67947` — docs(fern): port multi-source migration from main (#11044)

- Commit: `7674ddd67947babd8d2b7dba3bf3aa600de97dd7`
- Parent(s): `899dfa9c78ef2000e5e470ea68803d79addb42af`
- Author: `Ben Hamm <ben.hamm@gmail.com>`
- Authored: `2026-07-15T16:00:07-07:00`
- Committer: `Ben Hamm <ben.hamm@gmail.com>`
- Committed: `2026-07-15T16:00:07-07:00`
- Signature status: `verification error`

<details>
<summary>Full commit message</summary>

<pre>docs(fern): port multi-source migration from main (#11044)

Custom mdx-components (RecipeStyles) fail to resolve for pages under docs/
on this branch&#x27;s basepath-aware + fern 5.41.1 config — recipe pages render
unstyled with &#x27;Could not resolve ../../fern/components/RecipeStyles&#x27;.
Main fixed this in #11044; this ports the same two lines: multi-source: true
on the instance, drop experimental.basepath-aware, bump CLI to 5.57.0.
Keeps organization: ai-dynamo for this branch&#x27;s preview flow.

Co-Authored-By: Claude Fable 5 &lt;noreply@anthropic.com&gt;
Signed-off-by: Ben Hamm &lt;ben.hamm@gmail.com&gt;</pre>

</details>

### 23. `74da754361cf` — docs(recipes): restore Feature Benchmarks nav + sync content with main

- Commit: `74da754361cfde9260c2f4ce4e77f31f0d105b23`
- Parent(s): `7674ddd67947babd8d2b7dba3bf3aa600de97dd7`
- Author: `Ben Hamm <ben.hamm@gmail.com>`
- Authored: `2026-07-15T16:00:07-07:00`
- Committer: `Ben Hamm <ben.hamm@gmail.com>`
- Committed: `2026-07-15T16:00:07-07:00`
- Signature status: `verification error`

<details>
<summary>Full commit message</summary>

<pre>docs(recipes): restore Feature Benchmarks nav + sync content with main

The nav restructure dropped the Feature Benchmarks section entirely
(benchmarks/*.mdx orphaned; /benchmarks/browse 404s on the preview).
Restore it as a section under the Recipes tab — the evidence pages
behind the recipes, same page set + hidden-detail pattern as main.
Also sync 5 recipes/benchmarks files with main (Kimi benchmark retitle
#10830 + small fixes); this branch had no local edits to them.

Co-Authored-By: Claude Fable 5 &lt;noreply@anthropic.com&gt;
Signed-off-by: Ben Hamm &lt;ben.hamm@gmail.com&gt;</pre>

</details>

### 24. `9842ec774f31` — Merge pull request #5 from ai-dynamo/bhamm/docs-refactor-recipes-render

- Commit: `9842ec774f31483b5a55fbd96d00f86d7ce23c16`
- Parent(s): `31cf60a77646f335290903e534b938cac8eecd88 74da754361cfde9260c2f4ce4e77f31f0d105b23`
- Author: `Jonathan Tong <jt572@cornell.edu>`
- Authored: `2026-07-17T15:36:57-04:00`
- Committer: `GitHub <noreply@github.com>`
- Committed: `2026-07-17T15:36:57-04:00`
- Signature status: `verification error`

<details>
<summary>Full commit message</summary>

<pre>Merge pull request #5 from ai-dynamo/bhamm/docs-refactor-recipes-render

docs: fix recipe component rendering + restore Feature Benchmarks nav</pre>

</details>

### 25. `7663d0afb0ce` — backup: fern docs consolidation into docs/fern/ + CI path updates

- Commit: `7663d0afb0ce6b79cdb8020c9883b43845c76c76`
- Parent(s): `9842ec774f31483b5a55fbd96d00f86d7ce23c16`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-17T15:35:57-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-17T15:39:15-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>backup: fern docs consolidation into docs/fern/ + CI path updates

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 26. `22c486343fc2` — Work on local cli docs

- Commit: `22c486343fc2fc1d4f13e162fc161e3e33913523`
- Parent(s): `7663d0afb0ce6b79cdb8020c9883b43845c76c76`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-17T19:27:01-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-17T19:27:01-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>Work on local cli docs

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 27. `95448c7d2670` — docs(welcome): rework landing page — Get Started, community cards, events calendar

- Commit: `95448c7d267021a511cef5e22e0f93f00f94e706`
- Parent(s): `7663d0afb0ce6b79cdb8020c9883b43845c76c76`
- Author: `Harry Kim <harryk@nvidia.com>`
- Authored: `2026-07-17T14:39:20-07:00`
- Committer: `Harry Kim <harryk@nvidia.com>`
- Committed: `2026-07-17T14:39:20-07:00`
- Signature status: `unsigned`

<details>
<summary>Full commit message</summary>

<pre>docs(welcome): rework landing page — Get Started, community cards, events calendar

- Reorder sections: Get Started, Why Dynamo, Dynamo in Action, Community Events
- Get Started: Kubernetes Quickstart, Community Slack/WeChat (browser-language
  gated via LangGate), Community Calendar
- Rewrite Why Dynamo cards (performance, SW integration/interop, HW neutrality)
- Add Community Events section: EventsCalendar renders upcoming/past events from
  a build-time data module generated hourly from the public Google Calendar
  (.github/scripts/generate-events.js + .github/workflows/update-events.yml)

Signed-off-by: Harry Kim &lt;harryk@nvidia.com&gt;</pre>

</details>

### 28. `3b9bb00b95b7` — Merge pull request #6 from harryskim/welcome-community-events

- Commit: `3b9bb00b95b7e183b5798f3d24169b2900b1e66e`
- Parent(s): `7663d0afb0ce6b79cdb8020c9883b43845c76c76 95448c7d267021a511cef5e22e0f93f00f94e706`
- Author: `Jonathan Tong <jt572@cornell.edu>`
- Authored: `2026-07-17T19:21:59-04:00`
- Committer: `GitHub <noreply@github.com>`
- Committed: `2026-07-17T19:21:59-04:00`
- Signature status: `verification error`

<details>
<summary>Full commit message</summary>

<pre>Merge pull request #6 from harryskim/welcome-community-events

docs(welcome): rework landing page — Get Started, community cards, events calendar</pre>

</details>

### 29. `6b44e67d257b` — Merge branch 'dynamo-full-docs-refactor' of github.com:Jont828/dynamo into dynamo-full-docs-refactor

- Commit: `6b44e67d257b74faa4de0fab1150ae960da6df94`
- Parent(s): `22c486343fc2fc1d4f13e162fc161e3e33913523 3b9bb00b95b7e183b5798f3d24169b2900b1e66e`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-17T19:27:04-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-17T19:27:04-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>Merge branch &#x27;dynamo-full-docs-refactor&#x27; of github.com:Jont828/dynamo into dynamo-full-docs-refactor</pre>

</details>

### 30. `a283d1523938` — Fix

- Commit: `a283d1523938b78f07245aacb08ef5687ee3ec4b`
- Parent(s): `6b44e67d257b74faa4de0fab1150ae960da6df94`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-17T19:50:22-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-17T19:50:22-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>Fix

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 31. `c91e670a5ece` — Fix community slack card

- Commit: `c91e670a5ecea6a8a1fa9031e4f5149e888b3029`
- Parent(s): `a283d1523938b78f07245aacb08ef5687ee3ec4b`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-17T20:01:10-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-17T20:01:10-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>Fix community slack card

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 32. `814125e4c2c8` — Work on refactoring User Guide to use tab variants

- Commit: `814125e4c2c8b2c71a585806c85c7d2ab525aa03`
- Parent(s): `c91e670a5ecea6a8a1fa9031e4f5149e888b3029`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-21T17:19:46-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-21T17:19:46-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>Work on refactoring User Guide to use tab variants

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 33. `f62df6dfabe2` — docs(fern): fix User Guide variant sidebar section headings

- Commit: `f62df6dfabe2a001b10294b7357d5b060525359e`
- Parent(s): `814125e4c2c8b2c71a585806c85c7d2ab525aa03`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-21T17:57:12-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-21T17:57:12-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>docs(fern): fix User Guide variant sidebar section headings

Moving the User Guide tab to `variants:` nests its top-level sections one
level deeper, so Fern renders them with a lighter title weight than the
pre-variant fixed headers. Bump the level-1 title weight to 600, scoped via
`#fern-sidebar:has(.fern-variant-selector)` so only the variant tabs
(Kubernetes / Local CLI) are affected and all other tabs are left alone.
Chevron and click behaviour are preserved so sections stay collapsible.

Bump Fern to 5.76.0, whose sidebar DOM the selector targets.

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 34. `9d04f9e08814` — docs(observability): separate installation and operations

- Commit: `9d04f9e08814c12ce95e2638e6ad970e79cd8567`
- Parent(s): `f62df6dfabe2a001b10294b7357d5b060525359e`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-21T19:02:51-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-21T19:02:51-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>docs(observability): separate installation and operations

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 35. `e3a50d2fcfdc` — docs(fern): rewrite disaggregated serving as single-node tutorial

- Commit: `e3a50d2fcfdccf3af28bb607ea778775fe6e95ec`
- Parent(s): `9d04f9e08814c12ce95e2638e6ad970e79cd8567`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-21T19:22:36-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-21T19:22:36-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>docs(fern): rewrite disaggregated serving as single-node tutorial

Reframe the disaggregated serving page as an action-oriented, step-based
guide for writing a disaggregated DGD on a single multi-GPU node. Drop the
architecture diagram, add an agg-vs-disagg comparison, trim worker specs to
the minimal disagg-relevant fields, and move RDMA/cross-node setup to a
dedicated multi-node pointer step.

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 36. `c3aed471163b` — docs(reference): add local deployment examples

- Commit: `c3aed471163bc80499a39e5d2e9da121e4e59e53`
- Parent(s): `e3a50d2fcfdccf3af28bb607ea778775fe6e95ec`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-21T19:28:16-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-21T19:28:16-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>docs(reference): add local deployment examples

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 37. `67561f64d54b` — docs(cli): consolidate KV cache offloading guide

- Commit: `67561f64d54b4b72b7514dd6369a6aee68cb914e`
- Parent(s): `c3aed471163bc80499a39e5d2e9da121e4e59e53`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-21T20:12:04-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-21T20:12:04-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>docs(cli): consolidate KV cache offloading guide

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 38. `ebde3aa13690` — Rework landing hero

- Commit: `ebde3aa1369002b48c5199d3f47c5e852903ab7f`
- Parent(s): `67561f64d54b4b72b7514dd6369a6aee68cb914e`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-21T20:15:27-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-21T20:15:27-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>Rework landing hero

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 39. `fc4ffa68c5c5` — Refactor local cli guide pages for observability, deployment examples, add fern navigation skill, and keep working on landing page hero

- Commit: `fc4ffa68c5c50867a64aacb574b941ca6d8acf8c`
- Parent(s): `ebde3aa1369002b48c5199d3f47c5e852903ab7f`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-21T20:48:08-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-21T20:48:08-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>Refactor local cli guide pages for observability, deployment examples, add fern navigation skill, and keep working on landing page hero

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 40. `01222325966d` — Continue working on hero

- Commit: `01222325966dfb8daf351417c3e46bf6c958c9ab`
- Parent(s): `fc4ffa68c5c50867a64aacb574b941ca6d8acf8c`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-21T21:12:40-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-21T21:12:40-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>Continue working on hero

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 41. `8d845c6e429c` — Update calendar widget

- Commit: `8d845c6e429cfbc0aa6b7347cd9b9c9e0011bc81`
- Parent(s): `01222325966dfb8daf351417c3e46bf6c958c9ab`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-21T21:34:19-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-21T21:34:19-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>Update calendar widget

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 42. `10af11d7dcb3` — Try to add top lvl tab

- Commit: `10af11d7dcb334bb4629294e0ac56d17b0ae50a6`
- Parent(s): `8d845c6e429cfbc0aa6b7347cd9b9c9e0011bc81`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-22T11:49:09-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-22T11:49:09-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>Try to add top lvl tab

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 43. `be3fab58224f` — Try to add custom tab variant selector

- Commit: `be3fab58224f633c0ccc7ff9b6cce342a037efee`
- Parent(s): `10af11d7dcb334bb4629294e0ac56d17b0ae50a6`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-22T13:02:11-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-22T13:02:11-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>Try to add custom tab variant selector

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 44. `303267e3842d` — Refactor intro and fix icon for K8s/local selector

- Commit: `303267e3842df0be84d6b1dcb01c6fcbe7e91ad6`
- Parent(s): `be3fab58224f633c0ccc7ff9b6cce342a037efee`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-22T13:38:16-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-22T14:18:55-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>Refactor intro and fix icon for K8s/local selector

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 45. `8ad9dd228fb0` — Work on calendar widget, introduction, compatiblilty, and refactor references/recipes tabs

- Commit: `8ad9dd228fb0aba72a93c5cc0a99652d2c1ae45a`
- Parent(s): `303267e3842df0be84d6b1dcb01c6fcbe7e91ad6`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-22T16:44:13-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-22T16:44:16-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>Work on calendar widget, introduction, compatiblilty, and refactor references/recipes tabs

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 46. `a657bc32cccc` — Work on community and blog page

- Commit: `a657bc32cccc189bcf81658975d1f231ec507fd1`
- Parent(s): `8ad9dd228fb0aba72a93c5cc0a99652d2c1ae45a`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-22T19:55:49-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-22T19:55:49-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>Work on community and blog page

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 47. `611177259e9a` — Continue on community landing page

- Commit: `611177259e9a0d4998de2424a6db0a42b5cdc186`
- Parent(s): `a657bc32cccc189bcf81658975d1f231ec507fd1`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-22T21:21:13-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-22T21:21:13-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>Continue on community landing page

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 48. `de6e74c6bf65` — Fix home page and blog

- Commit: `de6e74c6bf65e384f19af2558f5a145a762bccfd`
- Parent(s): `611177259e9a0d4998de2424a6db0a42b5cdc186`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-23T15:12:20-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-23T15:12:20-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>Fix home page and blog

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 49. `9c11e46da693` — Revert navbar and page width

- Commit: `9c11e46da693fe539c1ab4ae31ac8ca84e14b639`
- Parent(s): `de6e74c6bf65e384f19af2558f5a145a762bccfd`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-23T16:40:33-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-23T16:40:33-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>Revert navbar and page width

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 50. `365f0ecba71a` — Work on blog

- Commit: `365f0ecba71a2c8e75de1e1a3609e3de58f492e9`
- Parent(s): `9c11e46da693fe539c1ab4ae31ac8ca84e14b639`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-23T16:50:08-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-23T16:50:08-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>Work on blog

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 51. `1e032cfa0b44` — Roll back blog and community pages to be more subtle

- Commit: `1e032cfa0b442ab389f8d5791c647517c4a71f2f`
- Parent(s): `365f0ecba71a2c8e75de1e1a3609e3de58f492e9`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-23T19:19:22-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-23T19:19:22-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>Roll back blog and community pages to be more subtle

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 52. `ee42f4496088` — Refactoring dynosim, edits to blog/community

- Commit: `ee42f44960883ef190ed04e1fb3576cca4719bb5`
- Parent(s): `1e032cfa0b442ab389f8d5791c647517c4a71f2f`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-23T20:37:01-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-23T20:37:01-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>Refactoring dynosim, edits to blog/community

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 53. `5746be07d138` — Work on DynoSim more

- Commit: `5746be07d1380cf30eaeb85efc5a0807dd990843`
- Parent(s): `ee42f44960883ef190ed04e1fb3576cca4719bb5`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-23T20:58:44-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-23T20:58:44-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>Work on DynoSim more

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 54. `37e9c7aebae6` — docs: refresh multimodal and diffusion guides

- Commit: `37e9c7aebae6e701be2da692e5f6c5dfcec7fd73`
- Parent(s): `5746be07d1380cf30eaeb85efc5a0807dd990843`
- Author: `akshatha-k <akshutk@gmail.com>`
- Authored: `2026-07-23T21:08:13-07:00`
- Committer: `akshatha-k <akshutk@gmail.com>`
- Committed: `2026-07-23T21:08:42-07:00`
- Signature status: `unsigned`

<details>
<summary>Full commit message</summary>

<pre>docs: refresh multimodal and diffusion guides

Restructure multimodal and diffusion content around clearer workflows, backend-specific guidance, and Fern components. Include the in-progress Agents documentation sync for continued follow-up.

Signed-off-by: akshatha-k &lt;akshutk@gmail.com&gt;</pre>

</details>

### 55. `b617015c0ed0` — docs: refine parsing and feature guides

- Commit: `b617015c0ed0de70931354d04164f33949547a26`
- Parent(s): `37e9c7aebae6e701be2da692e5f6c5dfcec7fd73`
- Author: `akshatha-k <akshutk@gmail.com>`
- Authored: `2026-07-24T00:39:52-07:00`
- Committer: `akshatha-k <akshutk@gmail.com>`
- Committed: `2026-07-24T00:39:52-07:00`
- Signature status: `unsigned`

<details>
<summary>Full commit message</summary>

<pre>docs: refine parsing and feature guides

Restructure parser documentation around guided workflows and on-demand model details, while updating Fastokens, navigation, and in-progress agent content.

Signed-off-by: akshatha-k &lt;akshutk@gmail.com&gt;</pre>

</details>

### 56. `48092531d29d` — docs(diffusion): consolidate modality guides

- Commit: `48092531d29df297bc9be0f6cb281d81e9d3e575`
- Parent(s): `b617015c0ed0de70931354d04164f33949547a26`
- Author: `akshatha-k <akshutk@gmail.com>`
- Authored: `2026-07-24T01:04:00-07:00`
- Committer: `akshatha-k <akshutk@gmail.com>`
- Committed: `2026-07-24T01:04:00-07:00`
- Signature status: `unsigned`

<details>
<summary>Full commit message</summary>

<pre>docs(diffusion): consolidate modality guides

Group backend-specific diffusion instructions into modality pages with tabs so users can compare implementations without navigating separate guides.

Signed-off-by: akshatha-k &lt;akshutk@gmail.com&gt;</pre>

</details>

### 57. `f7bd296ef3d7` — docs(lora): streamline adapter setup guide

- Commit: `f7bd296ef3d7431c3d2e7d4b3d9d53dc9e9b4275`
- Parent(s): `48092531d29df297bc9be0f6cb281d81e9d3e575`
- Author: `akshatha-k <akshutk@gmail.com>`
- Authored: `2026-07-24T12:58:31-07:00`
- Committer: `akshatha-k <akshutk@gmail.com>`
- Committed: `2026-07-24T12:58:31-07:00`
- Signature status: `unsigned`

<details>
<summary>Full commit message</summary>

<pre>docs(lora): streamline adapter setup guide

Fold configuration and deployment details into actionable Kubernetes and local workflows while keeping routing and troubleshooting details available on demand.

Signed-off-by: akshatha-k &lt;akshutk@gmail.com&gt;</pre>

</details>

### 58. `b708925a961c` — docs(navigation): flatten feature guide structure

- Commit: `b708925a961c786d60d46fd4f4e2bc66c1062cc3`
- Parent(s): `f7bd296ef3d7431c3d2e7d4b3d9d53dc9e9b4275`
- Author: `akshatha-k <akshutk@gmail.com>`
- Authored: `2026-07-24T13:18:15-07:00`
- Committer: `akshatha-k <akshutk@gmail.com>`
- Committed: `2026-07-24T13:18:15-07:00`
- Signature status: `unsigned`

<details>
<summary>Full commit message</summary>

<pre>docs(navigation): flatten feature guide structure

Expose feature families directly in navigation, preserve legacy URLs with redirects, and finish related link and callout cleanup.

Signed-off-by: akshatha-k &lt;akshutk@gmail.com&gt;</pre>

</details>

### 59. `92df7d3f8973` — Work on contributor guide sesction

- Commit: `92df7d3f89738712bd2a80dea687749a892a296e`
- Parent(s): `b708925a961c786d60d46fd4f4e2bc66c1062cc3`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-24T17:29:37-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-24T17:31:25-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>Work on contributor guide sesction

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>

### 60. `3f4405ed7017` — docs(reference): rebuild Compatibility + Release Artifacts as data-driven components, add Model Early Access Builds

- Commit: `3f4405ed701777bc1f3b203d2a18c4e51f174e5c`
- Parent(s): `611177259e9a0d4998de2424a6db0a42b5cdc186`
- Author: `Dan Gil <dagil@nvidia.com>`
- Authored: `2026-07-22T21:49:32-05:00`
- Committer: `Dan Gil <dagil@nvidia.com>`
- Committed: `2026-07-22T21:49:32-05:00`
- Signature status: `verification error`

<details>
<summary>Full commit message</summary>

<pre>docs(reference): rebuild Compatibility + Release Artifacts as data-driven components, add Model Early Access Builds

Rebuild the Reference-tab pages on the docs refactor as custom Fern
components backed by a single data module (components/releases.data.ts),
refreshed from main&#x27;s reference pages to v1.3.0 GA:

- CompatibilityHero, FeatureHeatmap (per-backend coverage scores,
  numbered caveat footnotes), BackendVersionMatrix (pin-diff highlights
  across the release history), CudaDriverMatrix (driver ladder showing
  the CUDA 12 cutoff at v1.3.0), ArtifactBrowser (CSS-only filter rail,
  click-to-copy tags), ReleaseTimeline, ModelEABuildCards (GA-path
  badges, per-tag coverage dots), shared ReferenceStyles vocabulary.
- New reference/model-early-access-builds.mdx page ported from main;
  added to the Reference tab nav and redirects.
- Fixed pre-existing broken support-matrix/feature-matrix links across
  backends/, features/, components/, getting-started/ and added
  /resources/* redirect sources matching the live site URLs.
- custom.js: clipboard binder for [data-dynref-copy] buttons.

Per-release bumps now touch only releases.data.ts. Design pass ran
against a 6-dimension rubric with fresh-context raters (round 1: 3.6/5,
round 2 after fixes: 4.61/5, ship gate passed).

Signed-off-by: Dan Gil &lt;dagil@nvidia.com&gt;</pre>

</details>

### 61. `7aa93a40bc27` — docs(reference): version-matrix layout refinements from review

- Commit: `7aa93a40bc27f4d334cd09a8fb76a80942746c6d`
- Parent(s): `3f4405ed701777bc1f3b203d2a18c4e51f174e5c`
- Author: `Dan Gil <dagil@nvidia.com>`
- Authored: `2026-07-22T23:30:06-05:00`
- Committer: `Dan Gil <dagil@nvidia.com>`
- Committed: `2026-07-22T23:30:06-05:00`
- Signature status: `verification error`

<details>
<summary>Full commit message</summary>

<pre>docs(reference): version-matrix layout refinements from review

- Kind pills (GA release / Patch / Early access / Model build) move to a
  dedicated Type column; stable releases are labeled GA release, not Minor
  (matrix + timeline).
- NIXL cell always renders three stacked per-backend rows.
- Per-release history accordions move inline under Backend Dependencies
  and CUDA &amp; Driver Requirements instead of a buried bottom section.

Signed-off-by: Dan Gil &lt;dagil@nvidia.com&gt;</pre>

</details>

### 62. `9fb5c2801e1c` — docs(reference): quality-gate fixes

- Commit: `9fb5c2801e1c93808d8b55155f22291c7687c16e`
- Parent(s): `7aa93a40bc27f4d334cd09a8fb76a80942746c6d`
- Author: `Dan Gil <dagil@nvidia.com>`
- Authored: `2026-07-22T23:39:44-05:00`
- Committer: `Dan Gil <dagil@nvidia.com>`
- Committed: `2026-07-22T23:39:44-05:00`
- Signature status: `verification error`

<details>
<summary>Full commit message</summary>

<pre>docs(reference): quality-gate fixes

- Add a legend note explaining the pin-change highlight in the version
  history table (unmarked pins are unchanged vs the previous release).
- Delete unused learnMoreLabel/learnMoreHref fields from releases.data.ts
  (nothing renders them; per-feature source links live in the MDX tabs).

Gate evidence: all nine component files pass tsc --strict; fern docs
broken-links reports zero errors in the three reference pages (189
pre-existing errors elsewhere on the branch); custom.js passes node
--check; key version pins audited against main&#x27;s reference pages.

Signed-off-by: Dan Gil &lt;dagil@nvidia.com&gt;</pre>

</details>

### 63. `b76a68044989` — docs(reference): residual-risk fixes from quality gate

- Commit: `b76a680449897176dfc3ecbf2366664becd2fba4`
- Parent(s): `9fb5c2801e1c93808d8b55155f22291c7687c16e`
- Author: `Dan Gil <dagil@nvidia.com>`
- Authored: `2026-07-23T01:06:27-05:00`
- Committer: `Dan Gil <dagil@nvidia.com>`
- Committed: `2026-07-23T01:06:27-05:00`
- Signature status: `verification error`

<details>
<summary>Full commit message</summary>

<pre>docs(reference): residual-risk fixes from quality gate

- Register multimodal-vllm/sglang/trtllm pages in a hidden nav section so
  Compatibility&#x27;s per-backend links can resolve.
- Darken light-mode accentPrimary to #538300 (2.41:1 -&gt; 4.55:1 contrast).
- Consolidate the green/amber tint families into --dynref-* CSS tokens
  (single source in ReferenceStyles; 54 literals replaced, 5 redundant
  .dark blocks removed).
- Cleanup: remove link-resolution test block.

Signed-off-by: Dan Gil &lt;dagil@nvidia.com&gt;</pre>

</details>

### 64. `b5e42313cc66` — docs(reference): release-notes mirror, Known Issues + Deprecations pages, artifacts streamline

- Commit: `b5e42313cc66052f203e07923464b6231002d6d3`
- Parent(s): `b76a680449897176dfc3ecbf2366664becd2fba4`
- Author: `Dan Gil <dagil@nvidia.com>`
- Authored: `2026-07-23T02:32:31-05:00`
- Committer: `Dan Gil <dagil@nvidia.com>`
- Committed: `2026-07-23T02:32:31-05:00`
- Signature status: `verification error`

<details>
<summary>Full commit message</summary>

<pre>docs(reference): release-notes mirror, Known Issues + Deprecations pages, artifacts streamline

Phase A of the reference-docs expansion:

- Release Notes: docs-native mirror of the GitHub release notes for
  v1.0.0+ (verbatim one-time ingestion; patches folded into base pages;
  contributors in accordions). Per-release ReleaseHeader (GA pill, stat
  tiles), ReleaseSummaryCards highlight grid, UpgradePanel migration strip.
- Known Issues + Deprecations pages split from the release bodies
  (area badges; Removed/Deprecated/Behavioral kind badges; Migrate
  guidance preserved); Known Artifact Issues table relocated.
- Release Artifacts streamline: timeline promoted + linked to release
  pages, release links in the browser header, EA explainer accordion.
- Compatibility hero: prominent backend names + per-backend CUDA chips.
- Agent readability: scripts/gen_llms_tables.py emits llms-only twins
  into the three componentized pages (idempotent markers).
- releases.data.ts notesHref; nav gains Release Notes section, Known
  Issues, Deprecations with pinned slugs.

Verification: fern dev renders all pages; broken-links 0 errors in the
nine reference pages (baseline 189 pre-existing); tsc --strict clean;
assembly verbatim audit zero drift.

Signed-off-by: Dan Gil &lt;dagil@nvidia.com&gt;</pre>

</details>

### 65. `82c06bf4d69a` — docs(reference): design-gate fixes for the release-notes suite

- Commit: `82c06bf4d69a545d1724cb35e3d3f5cdf089abd9`
- Parent(s): `b5e42313cc66052f203e07923464b6231002d6d3`
- Author: `Dan Gil <dagil@nvidia.com>`
- Authored: `2026-07-23T03:21:00-05:00`
- Committer: `Dan Gil <dagil@nvidia.com>`
- Committed: `2026-07-23T03:21:00-05:00`
- Signature status: `verification error`

<details>
<summary>Full commit message</summary>

<pre>docs(reference): design-gate fixes for the release-notes suite

Round-2 rater findings (gate was 4.54 with two dimensions below 4):

- ReleaseTimeline gains a notes variant: feature-voice summaries for GA
  releases (new notesSummary field) and no crates table on the Release
  Notes overview — ends the duplicated artifact-voice history.
- UpgradePanel drops the non-interactive from-version chips for plain
  muted text (single fromVersion API); no false affordance.
- Highlight-card area badges flattened to the uniform blue treatment.
- Hash deep-links now open the target accordion (hydration-aware
  handler in custom.js) and re-scroll to the anchor.
- Early-access pointer added to every release page&#x27;s cross-link note.
- Terminology unified on &#x27;GA release&#x27; (hero + version matrix).
- Ledger accordion titles unified to &#x27;vX.Y.Z — N entries&#x27; shape.

Skipped by design: trimming highlight-card copy (bodies are verbatim
mirrors of the GitHub release text).

tsc --strict clean; node --check clean; live-verified: overview voice
split, blue chips, accordion deep-links, GA release badges.

Signed-off-by: Dan Gil &lt;dagil@nvidia.com&gt;</pre>

</details>

### 66. `801b5aaea9f0` — docs(reference): quality-gate fixes for the release-notes increment

- Commit: `801b5aaea9f088b7a2d1fc0673368d7772607cb9`
- Parent(s): `82c06bf4d69a545d1724cb35e3d3f5cdf089abd9`
- Author: `Dan Gil <dagil@nvidia.com>`
- Authored: `2026-07-23T08:07:29-07:00`
- Committer: `Dan Gil <dagil@nvidia.com>`
- Committed: `2026-07-23T08:07:29-07:00`
- Signature status: `verification error`

<details>
<summary>Full commit message</summary>

<pre>docs(reference): quality-gate fixes for the release-notes increment

- releases.data.ts header now carries the real per-release bump
  checklist (data entry, new release page + ingestion-time counts,
  ledger sections, nav slug, twins regeneration) and points at the
  generator&#x27;s parser contract — the old &#x27;bump this file&#x27; claim
  understated the workflow and invited drift.
- Blue tint family consolidated into shared --dynref-blue-* tokens
  (badge--blue, chip--arch, summary-card area chips, upgrade-panel
  reading chips); redundant .dark blocks pruned — completes the
  green/amber token pattern from the earlier gate.

Gate evidence: tsc --strict clean across all 12 components;
gen_llms_tables.py --check passes (parser tolerates the new header,
twins fresh); blue tokens verified flipping in both modes live.

Signed-off-by: Dan Gil &lt;dagil@nvidia.com&gt;</pre>

</details>

### 67. `2e087184955e` — docs(reference): close all gate residuals

- Commit: `2e087184955e544a1742676ce4f73f8341202572`
- Parent(s): `801b5aaea9f088b7a2d1fc0673368d7772607cb9`
- Author: `Dan Gil <dagil@nvidia.com>`
- Authored: `2026-07-23T08:20:44-07:00`
- Committer: `Dan Gil <dagil@nvidia.com>`
- Committed: `2026-07-23T08:20:44-07:00`
- Signature status: `verification error`

<details>
<summary>Full commit message</summary>

<pre>docs(reference): close all gate residuals

- Per-release stats single-sourced in RELEASE_STATS (releases.data.ts):
  ReleaseHeader reads stats by version, UpgradePanel derives reading-list
  labels/hrefs from (version, kind) pairs, and the Known Issues and
  Deprecations accordion titles compute their counts via MDX expression
  attributes — count duplication across pages is gone. Live-verified
  byte-identical rendered counts.
- Remaining color families tokenized (--dynref-teal/orange/violet-*)
  across chips and artifact-browser glyphs; redundant .dark blocks pruned.
  Every dynref tint now has exactly one definition.
- New scripts/check_reference.sh: one-command gate (twins fresh +
  parser contract, custom.js parse, stale-link sweep, reference-scoped
  fern broken-links). Immediately caught 4 pre-existing broken links in
  reference/observability/operator-metrics.mdx (guide file deleted in an
  earlier refactor) — repointed to the surviving Kubernetes observability
  guide; site broken-links baseline drops 189 to 185.
- Prod .md-export verification attempted via fern generate --preview:
  blocked (publisher org membership) — lands with the PR publish
  pipeline; risk is one-sided (twins verified absent from human render).

check_reference.sh ALL CHECKS PASSED; tsc --strict clean across all 12
components; gen_llms_tables.py --check passes with RELEASE_STATS added.

Signed-off-by: Dan Gil &lt;dagil@nvidia.com&gt;</pre>

</details>

### 68. `9d24d9185868` — ci(fern-docs): fix preview build for the renamed digest posts + root cast asset

- Commit: `9d24d918586827114f50c200c5a36784df863968`
- Parent(s): `2e087184955e544a1742676ce4f73f8341202572`
- Author: `Dan Gil <dagil@nvidia.com>`
- Authored: `2026-07-23T09:50:45-07:00`
- Committer: `Dan Gil <dagil@nvidia.com>`
- Committed: `2026-07-23T09:50:45-07:00`
- Signature status: `verification error`

<details>
<summary>Full commit message</summary>

<pre>ci(fern-docs): fix preview build for the renamed digest posts + root cast asset

The docs-website version snapshots (fern/versions/v1.x.yml) reference the
digest posts as ../digest/&lt;post&gt;.md, but this branch renamed all six to
.mdx — the sync&#x27;s rm-rf+cp of digest/ then left those .md targets dangling
(12 fern check errors). And welcome.mdx points at ./assets/hero-demo-25.cast,
which the sync never lands in the root fern/assets/ (1 error).

- Sync docs/fern/assets/ into the docs-website root fern/assets/ (merge-copy,
  mirroring the blogs precedent so older versioned refs survive).
- Retarget ../digest/*.md -&gt; .mdx in the preserved version snapshots during
  sync (leaves ../blogs/*.md alone); a no-op once main commits the rewritten
  snapshots to docs-website.

Reproduced the full sync+check pipeline against a docs-website clone:
Found 13 errors -&gt; Found 0 errors (2 pre-existing warnings remain).

Signed-off-by: Dan Gil &lt;dagil@nvidia.com&gt;</pre>

</details>

### 69. `0b4b0210a05c` — docs(reference): Phase B upgrade tooling + Phase C agent surface

- Commit: `0b4b0210a05cb2401d43776e224f32caafb9012a`
- Parent(s): `9d24d918586827114f50c200c5a36784df863968`
- Author: `Dan Gil <dagil@nvidia.com>`
- Authored: `2026-07-23T08:42:53-07:00`
- Committer: `Dan Gil <dagil@nvidia.com>`
- Committed: `2026-07-23T10:44:15-07:00`
- Signature status: `verification error`

<details>
<summary>Full commit message</summary>

<pre>docs(reference): Phase B upgrade tooling + Phase C agent surface

Upgrade tooling (all CSS-only, data-derived from releases.data.ts):
- UpgradeSelector on Deprecations: pick your 1.x line, get the pill-to-
  pill dependency migration strip and the exact breaking-changes /
  known-issues reading list spanning every release to current. Reuses
  UpgradePanel internals via an export refactor (release-page panel
  renders identically).
- RunsWhereWizard on Compatibility: backend x CUDA-driver generation -&gt;
  qualifying releases with per-row driver floors; pull commands only for
  the current release; honest empty states (TRT-LLM x CUDA 12).
- PinnedEnvironment on Release Artifacts + the v1.3.0 notes page:
  backend-switched full setup script with a matching copy-all payload;
  TensorRT-LLM variant omits the wheel per policy.
- TagLookup on Release Artifacts: 13 published tags (stable + EA) -&gt;
  release/build detail cards with GA-path badges and ledger chips.

Agent surface:
- reference/releases-data.mdx: machine-readable mirror of the full data
  module, generated by gen_llms_tables.py between idempotent markers;
  hidden nav entry pins /reference/releases-data.
- assets/releases.json + assets/releases-atom.xml emitted by the same
  generator (deterministic dates, no now()); not site-servable by Fern
  (assets are CDN-rewritten only when referenced) - fetchable from the
  repo raw URL post-merge; noted on the Release Notes overview.
- Per-section llms.txt verified Fern-native (reference/llms.txt serves
  on the Fern origin; docs.nvidia.com proxy drops the suffix - flagged
  as an external follow-up). No docs.yml changes (the available keys
  would override the auto-generated indexes).

Gate: check_reference.sh ALL CHECKS PASSED; tsc --strict clean across
all 15 components; generator idempotent incl. new outputs; feed
xmllint-valid, JSON valid; all four surfaces live-verified.

Signed-off-by: Dan Gil &lt;dagil@nvidia.com&gt;</pre>

</details>

### 70. `f6b69b7c03cf` — docs(reference): consolidate release history into a Releases section, de-duplicate the timeline

- Commit: `f6b69b7c03cfa78b60d8b6374085ccce4484d6a9`
- Parent(s): `0b4b0210a05cb2401d43776e224f32caafb9012a`
- Author: `Dan Gil <dagil@nvidia.com>`
- Authored: `2026-07-23T10:32:08-07:00`
- Committer: `Dan Gil <dagil@nvidia.com>`
- Committed: `2026-07-23T10:44:15-07:00`
- Signature status: `verification error`

<details>
<summary>Full commit message</summary>

<pre>docs(reference): consolidate release history into a Releases section, de-duplicate the timeline

- Group the release pages under a &#x27;Releases&#x27; nav section in Reference
  (Release History + the four release notes pages + Known Issues +
  Deprecations); landing page retitled Release History.
- One canonical timeline: the feature-voice timeline on the Releases
  landing is now the only release history. Removed the duplicate
  artifact-voice timeline from Release Artifacts (it pointed at the same
  releases in a second voice).
- Extract the crates.io first-publication table into a standalone
  CratesFirstPublished component so it renders once, on Release Artifacts
  (its natural home as crate-publishing metadata), in an accordion.
- Release Artifacts now points to the Releases page for full history.

Signed-off-by: Dan Gil &lt;dagil@nvidia.com&gt;</pre>

</details>

### 71. `7fab6be86201` — docs(reference): design-rater polish on the Phase B surfaces

- Commit: `7fab6be86201f2caf6089b3e9fd4d91fad86b826`
- Parent(s): `f6b69b7c03cfa78b60d8b6374085ccce4484d6a9`
- Author: `Dan Gil <dagil@nvidia.com>`
- Authored: `2026-07-23T10:43:37-07:00`
- Committer: `Dan Gil <dagil@nvidia.com>`
- Committed: `2026-07-23T10:44:16-07:00`
- Signature status: `verification error`

<details>
<summary>Full commit message</summary>

<pre>docs(reference): design-rater polish on the Phase B surfaces

- RunsWhereWizard: fix the Current-badge overflow that overlapped the
  CUDA chip on the v1.3.0 row; add a data-derived empty-state hint.
- Keyboard accessibility across every radio-pill rail (wizard, upgrade
  selector, artifact browser, tag lookup, pinned environment): inputs
  are visually-hidden (dynref-vh) not display:none, with focus-visible
  rings on the active pill.
- PinnedEnvironment: break the helm line with a shell continuation so it
  no longer clips; copy-all payload stays byte-identical to the visible
  script.
- TagLookup: align the breaking/known-issue chips to the blue reading-chip
  treatment used on Deprecations.

tsc --strict clean; check_reference.sh ALL CHECKS PASSED.

Signed-off-by: Dan Gil &lt;dagil@nvidia.com&gt;</pre>

</details>

### 72. `696205fc9c8a` — docs(reference): structural quality-gate fixes

- Commit: `696205fc9c8a53fb173b94a7c83f68c83d3fe79c`
- Parent(s): `7fab6be86201f2caf6089b3e9fd4d91fad86b826`
- Author: `Dan Gil <dagil@nvidia.com>`
- Authored: `2026-07-23T11:18:28-07:00`
- Committer: `Dan Gil <dagil@nvidia.com>`
- Committed: `2026-07-23T11:18:28-07:00`
- Signature status: `verification error`

<details>
<summary>Full commit message</summary>

<pre>docs(reference): structural quality-gate fixes

From a fresh-context structural review of the Phase B/C surface:

- Generator: extract 7 shared section renderers (cuda/feature/artifact/
  known-issues/crates/platform/ea tables) so the machine-readable page and
  the llms-only twins compose them instead of duplicating byte-for-byte
  (-43 lines); add the three required-subscript exports to REQUIRED_EXPORTS
  so a missing one fails with the clean error; guard build_json against
  all-undated input like build_atom. Output proven byte-identical via
  golden diff; --check idempotent.
- ReleaseTimeline: drop the now-vestigial variant prop (only the notes
  voice is used since the crates table was extracted to CratesFirstPublished);
  fix the stale header comment; ReleaseTimeline() takes no args.
- UpgradePanel: drop the dead export on buildReadingChips (internal only).
- Deprecations: keep the v1.3.0 entry-count literal with a source-of-truth
  comment — a bare MDX {expression} at paragraph start breaks the page
  compile (verified: it 404&#x27;d), unlike the sibling accordion title={} attrs.

Reviewer&#x27;s count-drift finding on the release-notes prose was declined: those
counts are verbatim GitHub mirror (design invariant), and the derived values
render from RELEASE_STATS via ReleaseHeader; both verified consistent today.

tsc --strict clean; check_reference.sh ALL CHECKS PASSED; fern check 0 errors.

Signed-off-by: Dan Gil &lt;dagil@nvidia.com&gt;</pre>

</details>

### 73. `245534feae23` — docs(fern): align org and docs instance to dynamo main (nvidia, not ai-dynamo)

- Commit: `245534feae231fdf2199569d418dc05541dfe1c8`
- Parent(s): `696205fc9c8a53fb173b94a7c83f68c83d3fe79c`
- Author: `Dan Gil <dagil@nvidia.com>`
- Authored: `2026-07-23T11:23:06-07:00`
- Committer: `Dan Gil <dagil@nvidia.com>`
- Committed: `2026-07-23T11:23:06-07:00`
- Signature status: `verification error`

<details>
<summary>Full commit message</summary>

<pre>docs(fern): align org and docs instance to dynamo main (nvidia, not ai-dynamo)

The refactor branch had drifted to organization ai-dynamo and instance
ai-dynamo.docs.buildwithfern.com/dynamo, which no account can publish to.
dynamo main is authoritative: organization nvidia, instance
dynamo.docs.buildwithfern.com/dynamo with the docs.nvidia.com/dynamo custom
domains. Aligned both so previews and publishes route to the real org.

Verified: fern generate --docs --preview now publishes (267 pages).
Signed-off-by: Dan Gil &lt;dagil@nvidia.com&gt;</pre>

</details>

### 74. `3dacc4d1bf7e` — ci(fern): keep the Reference General variant shared across doc versions

- Commit: `3dacc4d1bf7e349d60317ef0203d962c58205c1f`
- Parent(s): `245534feae231fdf2199569d418dc05541dfe1c8`
- Author: `Dan Gil <dagil@nvidia.com>`
- Authored: `2026-07-23T16:55:53-07:00`
- Committer: `Dan Gil <dagil@nvidia.com>`
- Committed: `2026-07-23T16:55:53-07:00`
- Signature status: `verification error`

<details>
<summary>Full commit message</summary>

<pre>ci(fern): keep the Reference General variant shared across doc versions

The Reference tab&#x27;s General variant (Compatibility, Release Artifacts,
Releases, Known Issues, Deprecations, Model EA Builds, Glossary) is
cumulative metadata about releases, so version snapshots should render it
always-current instead of freezing a stale copy. This uses Fern&#x27;s native
shared-content model: every version&#x27;s nav references the same pages-dev
source files.

- release-version job: the pages-vX.Y.Z snapshot drops exactly the files
  the General variant references (derived from dev.yml, so versioned
  reference/ content like observability and runtime-config stays frozen),
  and the generated versions/vX.Y.Z.yml keeps that variant&#x27;s paths on
  ../pages-dev/. The Kubernetes API and Components variants stay on the
  frozen snapshot.
- sync job: after each main push, the General variant&#x27;s nav block from
  dev.yml is propagated into every post-rework versions/v*.yml, so pages
  added to the shared reference (e.g. a new release-notes page) appear in
  every version dropdown. Pre-rework snapshots have no such variant and
  are skipped untouched.

Validated by simulating both jobs locally against the current
docs-website tip (cutting a v1.4.0): the generated version yml carries
the intended shared/frozen path split, pre-rework version files are
byte-untouched, and fern check reports an identical error set to a
control run with the unmodified logic (all pre-existing, none in
reference/).

Signed-off-by: Dan Gil &lt;dagil@nvidia.com&gt;</pre>

</details>

### 75. `50a0a36e3e12` — docs(reference): fix hrefs broken by the Releases-section restructure + gate the class

- Commit: `50a0a36e3e12d48e9e4e45995d86b1d74a381ade`
- Parent(s): `3dacc4d1bf7e349d60317ef0203d962c58205c1f`
- Author: `Dan Gil <dagil@nvidia.com>`
- Authored: `2026-07-23T17:20:16-07:00`
- Committer: `Dan Gil <dagil@nvidia.com>`
- Committed: `2026-07-23T17:20:16-07:00`
- Signature status: `verification error`

<details>
<summary>Full commit message</summary>

<pre>docs(reference): fix hrefs broken by the Releases-section restructure + gate the class

Moving Known Issues, Deprecations, and the release-notes pages under the
Releases nav section changed their URLs to /reference/releases/..., but
the absolute hrefs hardcoded in components and the data module kept the
old shape and 404 on the live preview (verified: /dynamo/dev/reference/
deprecations 404s, /dynamo/dev/reference/releases/deprecations 200s).

- Fix the six component href sites (ReleaseHeader, UpgradePanel,
  TagLookup breaking/known-issues chips), the ten notesHref values in
  releases.data.ts, and the Atom feed index URL in gen_llms_tables.py
  (now the release-history landing page); regenerate releases-data.mdx,
  releases.json, and releases-atom.xml from the fixed data.
- check_reference.sh gains step 4/5: every absolute
  /dynamo/dev/reference/... href in components, the data module,
  generated assets, and reference pages must match a URL derived from
  the index.yml Reference General variant (explicit slug or kebab-cased
  title, section slugs included). fern broken-links cannot see hrefs in
  TSX/JSON, which is exactly how this class escaped. Negative-tested: a
  reintroduced stale href fails the gate with file:line.

Full gate passes: twins fresh, custom.js parses, no stale matrix links,
12 published reference URLs all matched, 0 broken links in reference/.

Signed-off-by: Dan Gil &lt;dagil@nvidia.com&gt;</pre>

</details>

### 76. `22edc3b487a8` — ci(fern): warn when the shared Reference variant selectors stop matching

- Commit: `22edc3b487a816415a8e283ea069e45895718290`
- Parent(s): `50a0a36e3e12d48e9e4e45995d86b1d74a381ade`
- Author: `Dan Gil <dagil@nvidia.com>`
- Authored: `2026-07-23T17:20:32-07:00`
- Committer: `Dan Gil <dagil@nvidia.com>`
- Committed: `2026-07-23T17:20:32-07:00`
- Signature status: `verification error`

<details>
<summary>Full commit message</summary>

<pre>ci(fern): warn when the shared Reference variant selectors stop matching

If the reference tab or its General variant is renamed, the shared-
reference exclusion and nav revert in the release job silently no-op and
new version snapshots quietly freeze their own reference copy — the site
keeps working, so nothing surfaces the regression. Emit a workflow
warning annotation so the rename gets caught on the first tag cut.

Signed-off-by: Dan Gil &lt;dagil@nvidia.com&gt;</pre>

</details>

### 77. `80e97296eef9` — ci(fern): fix docs-website composition for component doc pages and deploy-manifest embeds

- Commit: `80e97296eef918b8b7cf421bdebcec61c41b4fa8`
- Parent(s): `22edc3b487a816415a8e283ea069e45895718290`
- Author: `Dan Gil <dagil@nvidia.com>`
- Authored: `2026-07-23T17:43:13-07:00`
- Committer: `Dan Gil <dagil@nvidia.com>`
- Committed: `2026-07-23T17:43:13-07:00`
- Signature status: `verification error`

<details>
<summary>Full commit message</summary>

<pre>ci(fern): fix docs-website composition for component doc pages and deploy-manifest embeds

Two pre-existing missing-file classes would fail the first post-merge
publish to docs-website (96 fern check errors on the composed tree; the
preview never hits them because it builds from the source layout):

- Component doc pages (64 errors): the nav references .md/.mdx pages
  under components/ from the Developer Guide tab and the Reference tab&#x27;s
  Components variant, but the sync rsync excludes components/ wholesale
  (so React .tsx sync separately to fern/components/ for mdx-components
  resolution). Add a second targeted rsync that copies only .md/.mdx
  under components/ into pages-dev/components/ — the .tsx all sit at the
  components/ root and the 38 doc pages all sit in subdirectories, so
  nothing duplicates, and the release job&#x27;s snapshot keeps freezing the
  config references per version as intended.

- Backend deploy manifests (32 errors): templates/{vllm,sglang,trtllm}
  .mdx embed repo files via &lt;Code src=&quot;../../../examples/backends/
  &lt;engine&gt;/deploy/*.yaml&quot;&gt;, which resolve against the repo root in the
  source layout but have no counterpart on docs-website. Sync just the
  referenced backends/*/deploy/ subtrees to the branch root; the same
  relative depth resolves from pages-dev and from version snapshots.

Validated by simulating the sync and release jobs against the current
docs-website tip with a v1.4.0 cut: fern check on the composed tree goes
from 96 errors to 0, no .tsx lands in pages-dev, all 38 doc pages and 86
deploy manifests sync, and the frozen snapshot carries the doc pages.
The sync job&#x27;s git add -A picks up both new trees unchanged.

Signed-off-by: Dan Gil &lt;dagil@nvidia.com&gt;</pre>

</details>

### 78. `3ad1d7c23b08` — test(fern): commit the docs-website composition regression harness

- Commit: `3ad1d7c23b08b80560e271ca05ca810cff477780`
- Parent(s): `80e97296eef918b8b7cf421bdebcec61c41b4fa8`
- Author: `Dan Gil <dagil@nvidia.com>`
- Authored: `2026-07-23T18:20:15-07:00`
- Committer: `Dan Gil <dagil@nvidia.com>`
- Committed: `2026-07-23T18:20:15-07:00`
- Signature status: `verification error`

<details>
<summary>Full commit message</summary>

<pre>test(fern): commit the docs-website composition regression harness

The fern-docs.yml sync and release-version jobs only execute on main
pushes and tag cuts, so composition changes (rsync scopes, nav path
transforms, the shared-Reference machinery) were validated by an
ephemeral scratchpad script that the next contributor could not rerun.

Land it as docs/fern/scripts/simulate_docs_website.sh: replays both jobs
against a temporary worktree of the local docs-website branch and
asserts the invariants — fern check reports 0 errors on the composed
tree, the generated version yml keeps the Reference General variant
shared while Kubernetes API and Components stay frozen, the snapshot
drops exactly the shared files, no .tsx leaks into pages-dev, pre-rework
version files gain no shared-reference pointers, and a page added to the
shared reference after a version is cut propagates into that version&#x27;s
nav (round two). Portable: derives paths from the repo root, no
hardcoded scratch dirs, perl instead of sed -i, python 3.10+ detection,
fern check degrades to a warning when the CLI is absent. shellcheck
clean; full run passes all 7 assertions against the current
docs-website tip.

The workflow header now points at the harness so it gets rerun before
the next composition change.

Signed-off-by: Dan Gil &lt;dagil@nvidia.com&gt;</pre>

</details>

### 79. `69b95917c7f5` — docs(reference): add UCX version per release, coupled to the NIXL pins

- Commit: `69b95917c7f5f04de78470c838caf509b673429c`
- Parent(s): `3ad1d7c23b08b80560e271ca05ca810cff477780`
- Author: `Dan Gil <dagil@nvidia.com>`
- Authored: `2026-07-24T11:41:37-07:00`
- Committer: `Dan Gil <dagil@nvidia.com>`
- Committed: `2026-07-24T11:41:37-07:00`
- Signature status: `verification error`

<details>
<summary>Full commit message</summary>

<pre>docs(reference): add UCX version per release, coupled to the NIXL pins

UCX ships with each release&#x27;s NIXL builds, so surface it wherever the
NIXL pins already render. Values are transcribed from the releases&#x27; own
Key Dependencies tables: 1.20 (v1.1.0), 1.20.0 (v1.2.0), 1.20.x
(v1.3.0). v1.0.0 and the patch releases never stated one, so they render
a dash rather than an inferred value.

- releases.data.ts: optional ucx field on Release (+ bump checklist)
- BackendVersionMatrix: UCX column after NIXL, with the same
  changed-since-previous-release highlight; the previous-release helper
  now returns the Release so non-pin fields can diff too
- UpgradePanel/UpgradeSelector: UCX row in the migration strip directly
  under the NIXL rows, skipped when either side has no stated value
- gen_llms_tables.py: UCX column in both generated pin tables;
  releases.json picks the field up via the existing passthrough

Twins regenerated and idempotent; check_reference.sh full pass.

Signed-off-by: Dan Gil &lt;dagil@nvidia.com&gt;</pre>

</details>

### 80. `785567915663` — docs(reference): merge v1.3.0 reference components

- Commit: `785567915663fbb753a16cdd3faea243fcdbcf73`
- Parent(s): `92df7d3f89738712bd2a80dea687749a892a296e 69b95917c7f5f04de78470c838caf509b673429c`
- Author: `Jont828 <jt572@cornell.edu>`
- Authored: `2026-07-24T18:05:54-04:00`
- Committer: `Jont828 <jt572@cornell.edu>`
- Committed: `2026-07-24T18:05:54-04:00`
- Signature status: `good (jt572@cornell.edu)`

<details>
<summary>Full commit message</summary>

<pre>docs(reference): merge v1.3.0 reference components

Merge upstream/dagil/reference-components-v1.3.0 into the docs refactor.\n\nResolve the navigation and workflow conflicts for the versions-only site layout, retain only the Reference click-to-copy and accordion deep-link behavior in custom.js, and keep the refactored LoRA and use-case navigation.

Signed-off-by: Jont828 &lt;jt572@cornell.edu&gt;</pre>

</details>
