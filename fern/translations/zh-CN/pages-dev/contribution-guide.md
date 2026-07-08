---
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: 贡献指南
subtitle: 如何为 Dynamo 做贡献
max-toc-depth: 3
---

Dynamo 是一个开源分布式推理平台，由不断壮大的贡献者社区共同构建。该项目采用 [Apache 2.0](https://github.com/ai-dynamo/dynamo/blob/main/LICENSE) 许可证，欢迎各种规模的贡献 -- 从修正错别字到开发重要功能都包括在内。社区贡献已经塑造了 Dynamo 的核心领域，包括后端集成、文档、部署工具和性能改进。

Dynamo 拥有 200 多位外部贡献者、220 多个已合并的社区 PR，并且每月都有新贡献者加入，是增长最快的开源推理项目之一。欢迎查看我们的[提交活动](https://github.com/ai-dynamo/dynamo/graphs/commit-activity)和 [GitHub stars](https://github.com/ai-dynamo/dynamo)。本指南将帮助你开始贡献。

加入社区：

- [CNCF Slack (`#ai-dynamo`)](https://communityinviter.com/apps/cloud-native/cncf) -- 加入 CNCF Slack，并在 `#ai-dynamo` 中找到我们
- [Discord](https://discord.gg/D92uqZRjCZ)
- [GitHub Discussions](https://github.com/ai-dynamo/dynamo/discussions)
- [设计提案](https://github.com/ai-dynamo/enhancements) -- 重大功能的 RFC
- [Office Hours](https://www.youtube.com/playlist?list=PL5B692fm6--tgryKu94h2Zb7jTFM3Go4X) -- 双周会议
- [社区会议](https://docs.google.com/document/d/1uR8xD_hlYGwV6QspvSc36k1H-wo1BUcVmFbHH9xlXd8/view) ([Youtube](https://www.youtube.com/@ai-dynamo-community)) -- 每周（Wed 10:30 AM PT）开发者社区会议
- [Dynamo Day 录像](https://nvevents.nvidia.com/dynamoday) -- 来自生产用户的深入分享

## TL;DR

面向有经验的贡献者：

1. Fork 并克隆仓库
2. 对于 ≥100 行的变更或新功能，请先[创建 issue](https://github.com/ai-dynamo/dynamo/issues/new?template=contribution_request.yml)
3. 创建分支：`git checkout -b yourname/fix-router-timeout`
4. 修改代码，运行 `pre-commit run`
5. 使用 DCO sign-off 提交：`git commit -s -m "fix: description"`
6. 打开一个面向 `main` 的 PR

---

## 贡献方式

### 报告 Bug

发现哪里坏了吗？请[提交 bug 报告](https://github.com/ai-dynamo/dynamo/issues/new?template=bug_report.yml)，并包含：

- 复现步骤
- 预期行为与实际行为
- 环境详情（OS、GPU、Python 版本、Dynamo 版本）

### 改进文档

我们始终欢迎文档改进：

- 修正错别字或不清楚的说明
- 添加示例或教程
- 改进 API 文档

较小的文档修复可以不创建 issue，直接作为 PR 提交。

### 提议功能

有想法吗？请[创建功能请求](https://github.com/ai-dynamo/dynamo/issues/new?template=feature_request.yml)，在实现前与维护者讨论。

### 贡献代码

准备写代码了吗？请参阅下方的[贡献流程](#贡献流程)部分。

### 帮助社区

并非所有贡献都是代码。你还可以：

- 在 [Discord](https://discord.gg/D92uqZRjCZ) 或 [CNCF Slack](https://communityinviter.com/apps/cloud-native/cncf) 的 `#ai-dynamo` 频道中回答问题
- 审阅 pull request
- 分享你如何使用 Dynamo -- 博客文章、演讲或社交媒体都可以
- 为[仓库](https://github.com/ai-dynamo/dynamo)点 star

---

## 入门

### 查找 Issue

浏览[开放 issue](https://github.com/ai-dynamo/dynamo/issues)，或查找：

| Issue 类型 | 说明 |
|------------|------|
| [Good First Issues](https://github.com/ai-dynamo/dynamo/labels/good-first-issue) | 适合初学者，并带有指导 |
| [Help Wanted](https://github.com/ai-dynamo/dynamo/labels/help-wanted) | 欢迎社区贡献 |

### Fork 并克隆

1. 在 GitHub 上 [Fork 仓库](https://github.com/ai-dynamo/dynamo/fork)
2. 克隆你的 fork：

```bash
git clone https://github.com/YOUR-USERNAME/dynamo.git
cd dynamo
git remote add upstream https://github.com/ai-dynamo/dynamo.git
```

### 从源码构建

> [!TIP]
> 完整构建说明包含在下方。展开折叠区以设置本地开发环境。

<details>
<summary>展开构建说明</summary>

#### 1. 安装系统库

**Ubuntu:**

```bash
sudo apt install -y build-essential libhwloc-dev libudev-dev pkg-config libclang-dev protobuf-compiler python3-dev cmake
```

**macOS:**

```bash
# Install Homebrew if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install cmake protobuf

# Verify Metal is accessible
xcrun -sdk macosx metal
```

#### 2. 安装 Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

#### 3. 创建 Python 虚拟环境

如果你还没有安装 [uv](https://docs.astral.sh/uv/#installation)，请先安装：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

创建并激活虚拟环境：

```bash
uv venv .venv
source .venv/bin/activate
```

#### 4. 安装构建工具

```bash
uv pip install pip 'maturin[patchelf]'
```

[Maturin](https://github.com/PyO3/maturin) 是 Rust-Python 绑定的构建工具。`patchelf` extra 让 maturin 能在构建期间修补原生扩展库路径。

#### 5. 构建 Rust 绑定

```bash
cd lib/bindings/python
maturin develop --uv
```

#### 6. 安装 GPU Memory Service

```bash
# Return to project root
cd "$(git rev-parse --show-toplevel)"
uv pip install -e lib/gpu_memory_service
```

#### 7. 安装 Wheel

```bash
uv pip install -e .
```

#### 8. 验证构建

```bash
python3 -m dynamo.frontend --help
```

> [!TIP]
> VSCode 和 Cursor 用户可以使用 [`.devcontainer`](https://github.com/ai-dynamo/dynamo/tree/main/.devcontainer) 文件夹获得预配置的开发环境。详情请参阅 [devcontainer README](https://github.com/ai-dynamo/dynamo/blob/main/.devcontainer/README.md)。

</details>

### 设置 Pre-commit Hooks

```bash
uv pip install pre-commit
pre-commit install
```

你已经设置好了！保持好奇 -- 探索代码库，试用[示例](https://github.com/ai-dynamo/dynamo/tree/main/examples)，看看各个部分如何协同工作。准备好后，可以从 [Good First Issues](https://github.com/ai-dynamo/dynamo/labels/good-first-issue) 看板中挑选一个 issue，或继续阅读完整贡献流程。

---

## 贡献流程

贡献流程取决于变更的大小和范围。即使不是必需，创建 issue 也是一个很好的开端，可以在投入时间编写 PR 前先与 Dynamo 维护者交流。

| 大小 | 变更行数 | 示例 | 你需要做什么 |
|------|----------|------|--------------|
| **XS** | 1–10 | 修正错别字、调整配置 | 直接提交 PR |
| **S** | 10–100 | 小型 bug 修复、文档改进、聚焦的重构 | 直接提交 PR |
| **M** | 100–200 | 添加功能、中等规模重构 | 先[创建 issue](https://github.com/ai-dynamo/dynamo/issues/new?template=contribution_request.yml) |
| **L** | 200–500 | 多文件功能、新组件 | 先[创建 issue](https://github.com/ai-dynamo/dynamo/issues/new?template=contribution_request.yml) |
| **XL** | 500–1000 | 重要功能、跨组件变更 | 先[创建 issue](https://github.com/ai-dynamo/dynamo/issues/new?template=contribution_request.yml) |
| **XXL** | 1000+ | 架构变更 | 需要一个 [DEP](https://github.com/ai-dynamo/enhancements) |

**小型变更（少于 100 行）：** 直接提交 PR -- 不需要 issue。这包括错别字、简单 bug 修复和格式调整。如果你的 PR 处理的是已有且已批准的 issue，请使用 "Fixes #123" 链接它。

**较大变更（≥100 行）：** 请先[创建 Contribution Request](https://github.com/ai-dynamo/dynamo/issues/new?template=contribution_request.yml) issue，并等待 `approved-for-pr` 标签后再提交 PR。

**架构变更：** 影响多个组件、引入或修改公共 API、改变通信平面架构，或影响后端集成契约的变更，都需要一个 [Dynamo Enhancement Proposal (DEP)](https://github.com/ai-dynamo/enhancements)。在开始实现前，请在 [`ai-dynamo/enhancements`](https://github.com/ai-dynamo/enhancements) 仓库中创建 DEP。

### 提交 Pull Request

1. **创建 GitHub Issue**（如果需要）— [创建 Contribution Request](https://github.com/ai-dynamo/dynamo/issues/new?template=contribution_request.yml)，说明你要解决的问题、建议方案、预计 PR 大小以及受影响文件。

2. **获得批准** — 等待维护者审阅并添加 `approved-for-pr` 标签。

3. **提交 Pull Request** — [创建 PR](https://github.com/ai-dynamo/dynamo/compare)，并使用 GitHub 关键字引用该 issue（例如 "Fixes #123"）。

4. **处理 Code Rabbit Review** — 回复自动化 Code Rabbit 建议，包括 nitpick。

5. **触发 CI 测试** — 对于外部贡献者，维护者必须评论 `/ok to test COMMIT-ID` 才能运行完整 CI 套件，其中 `COMMIT-ID` 是你最新提交的短 SHA。请求人工审阅前，请修复所有失败的测试。

6. **请求审阅** — 将批准你 issue 的人员添加为 reviewer。根据修改的文件，查看 [CODEOWNERS](https://github.com/ai-dynamo/dynamo/blob/main/CODEOWNERS) 了解必需 approver。

> [!IMPORTANT]
> **AI 生成代码：** 虽然我们鼓励使用 AI 工具，但你必须完全理解 PR 中的每一处变更。如果无法解释提交的代码，PR 将被拒绝。

### 分支命名

使用描述性的分支名，标识你本人和变更内容：

```text
yourname/fix-description
```

示例：

```text
jsmith/fix-router-timeout
jsmith/add-lora-support
```

---

## 代码风格与质量

维护者会根据代码风格、测试覆盖率、架构一致性和对审阅反馈的响应情况来评估贡献质量。持续的高质量贡献是在项目中建立信任的基础。

### Pre-commit Hooks

所有 PR 都会通过 [pre-commit hooks](https://github.com/ai-dynamo/dynamo/blob/main/.pre-commit-config.yaml) 检查。在[安装 pre-commit](#设置-pre-commit-hooks) 后，在本地运行检查：

```bash
pre-commit run --all-files
```

### Commit Message 约定

使用 [conventional commit](https://www.conventionalcommits.org/) 前缀：

| 前缀 | 用途 |
|------|------|
| `feat:` | 新功能 |
| `fix:` | Bug 修复 |
| `docs:` | 文档变更 |
| `refactor:` | 代码重构（无行为变更） |
| `test:` | 添加或更新测试 |
| `chore:` | 维护、依赖更新 |
| `ci:` | CI/CD 变更 |
| `perf:` | 性能改进 |

示例：

```text
feat(router): add weighted load balancing
fix(frontend): resolve streaming timeout on large responses
docs: update quickstart for macOS users
test(planner): add unit tests for scaling policy
```

### 语言约定

| 语言 | 风格指南 | 格式化工具 |
|------|----------|------------|
| **Python** | [PEP 8](https://peps.python.org/pep-0008/) | `black`, `ruff` |
| **Rust** | [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/) | `cargo fmt`, `cargo clippy` |
| **Go** | [Effective Go](https://go.dev/doc/effective_go) | `gofmt` |

### 测试

提交 PR 前请运行测试套件：

```bash
# Run all tests
pytest tests/

# Run unit tests only
pytest -m unit tests/

# Run a specific test file
pytest -s -v tests/test_example.py
```

对于 Rust 组件：

```bash
cargo test
```

对于 Kubernetes operator（Go）：

```bash
cd deploy/operator
go test ./... -v
```

### 通用准则

- 保持 PR 聚焦 -- 每个 PR 只处理一个关注点
- 编写干净、文档完善、未来贡献者能够理解的代码
- 为新功能和 bug 修复包含测试
- 确保构建干净（无警告或错误）
- 所有测试必须通过
- 不要提交被注释掉的代码
- 及时且建设性地回应审阅反馈

### 在本地运行 GitHub Actions

使用 [act](https://nektosact.com/) 在本地运行 workflow：

```bash
act -j pre-merge-rust
```

或者使用 [GitHub Local Actions](https://marketplace.visualstudio.com/items?itemName=SanjulaGanepola.github-local-actions) VS Code 扩展。

---

## 预期事项

### 状态标签

| 状态 | 含义 |
|------|------|
| `needs-triage` | 我们正在审阅你的 issue |
| `needs-info` | 我们需要你提供更多详情 |
| `approved-for-pr` | 已可实现 — 提交 PR |
| `in-progress` | 有人正在处理 |
| `blocked` | 正在等待外部依赖 |

### 响应时间

我们的目标是：

- 在几个工作日内**回复**新的 issue
- 在一周内**分流处理**高优先级 issue

30 天无活动的 issue 可能会被自动关闭（可以重新打开）。

### 审阅流程

提交 PR 并完成[提交 Pull Request](#提交-pull-request)中的步骤后：

1. Reviewer 会提供反馈 -- 请在合理时间内回复所有评论
2. 如果要求修改，请完成修改并提醒 reviewer 重新审阅
3. 如果你的 PR 在 7 天内仍未被审阅，可以提醒 reviewer 或留言

### Good First Issues

标记为 `good-first-issue` 的 issue 适合新贡献者。我们会为这些 issue 提供额外指导 -- 请在 issue 描述中查找清晰的验收标准和建议方案。

---

## DCO 与许可

### Developer Certificate of Origin

Dynamo 要求所有贡献都使用 [Developer Certificate of Origin (DCO)](https://developercertificate.org/) 进行签署。这证明你有权根据项目的 [Apache 2.0 license](https://github.com/ai-dynamo/dynamo/blob/main/LICENSE) 提交你的贡献。

每个提交都必须包含 sign-off 行：

```text
Signed-off-by: Jane Smith &lt;jane.smith@email.com&gt;
```

使用 `-s` 标志可以自动添加：

```bash
git commit -s -m "fix: your descriptive message"
```

**要求：**

- 使用真实姓名（不接受化名或匿名贡献）
- 你的 `user.name` 和 `user.email` 必须在 git 中配置

**DCO 检查失败？** 请参阅我们的 [DCO 故障排除指南](https://github.com/ai-dynamo/dynamo/blob/main/DCO.md)，按步骤修复。

### 许可证

通过贡献，你同意你的贡献将依据 [Apache 2.0 License](https://github.com/ai-dynamo/dynamo/blob/main/LICENSE) 授权。

---

## 行为准则

我们致力于提供一个欢迎且包容的环境。所有参与者都应遵守我们的 [Code of Conduct](https://github.com/ai-dynamo/dynamo/blob/main/CODE_OF_CONDUCT.md)。

---

## 安全

如果你发现安全漏洞，请遵循我们的 [Security Policy](https://github.com/ai-dynamo/dynamo/blob/main/SECURITY.md) 中的说明。请不要为安全漏洞创建公开 issue。

---

## 获取帮助

- **CNCF Slack**: [加入 CNCF Slack](https://communityinviter.com/apps/cloud-native/cncf)，并在 `#ai-dynamo` 中找到我们
- **Discord**: [加入我们的社区](https://discord.gg/D92uqZRjCZ)
- **Discussions**: [GitHub Discussions](https://github.com/ai-dynamo/dynamo/discussions)
- **设计提案**: [重大功能的 RFC](https://github.com/ai-dynamo/enhancements)
- **Office Hours**: [双周会议](https://www.youtube.com/playlist?list=PL5B692fm6--tgryKu94h2Zb7jTFM3Go4X)
- **社区会议**: [每周（Wed 10:30 AM PT）开发者社区会议](https://docs.google.com/document/d/1uR8xD_hlYGwV6QspvSc36k1H-wo1BUcVmFbHH9xlXd8/view) ([Youtube](https://www.youtube.com/@ai-dynamo-community))
- **Dynamo Day 录像**: [来自生产用户的深入分享](https://nvevents.nvidia.com/dynamoday)
- **Documentation**: [docs.nvidia.com/dynamo](https://docs.nvidia.com/dynamo/)

感谢你为 Dynamo 做贡献！
