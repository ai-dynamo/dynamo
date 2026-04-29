---
name: dynamo-bug
description: File a GitHub bug issue against ai-dynamo/dynamo using context from the current conversation.
user-invocable: true
---

# File a Dynamo Bug Issue

Use the current conversation context to file a well-structured bug report against `ai-dynamo/dynamo` via the `gh` CLI.

## Instructions

1. **Gather context from the conversation.** Review what the user has been working on, the problem encountered, error messages, logs, stack traces, and any reproduction steps already discussed. If critical details are missing, ask the user briefly — but prefer inferring from conversation context over asking.

2. **Collect environment info.** Determine whether the user is running in a **Kubernetes** or **local development** environment based on conversation context. Then gather the relevant environment details:

   For **Kubernetes** environments:
   - K8s version / distribution (e.g., EKS, GKE, kind)
   - Dynamo runtime version / container image tag
   - Node OS and CPU architecture
   - CUDA version and GPU architecture (if applicable)
   - Python version (if applicable)
   - Helm chart version or manifest details

   For **local development** environments:
   - OS and version
   - Dynamo runtime version
   - CPU architecture
   - CUDA version and GPU architecture (if applicable)
   - Python version

   Use shell commands to auto-detect what you can (e.g., `uname -m`, `python3 --version`, `nvidia-smi`, `kubectl version`). Fill in what's available and mark unknowns as "N/A".

3. **Draft the issue** using this template and present it to the user for review before filing:

   ```
   **Describe the Bug**
   <clear, concise description>

   **Steps to Reproduce**
   1. ...
   2. ...
   <!-- Include relevant manifests or public container references if applicable -->

   **Expected Behavior**
   <what should have happened>

   **Actual Behavior**
   <what actually happened — include error messages, logs, or stack traces>

   **Environment**
   - **OS:** ...
   - **Dynamo Runtime Version:** ...
   - **CPU Architecture:** ...
   - **CUDA Version:** ...
   - **GPU Architecture:** ...
   - **Python Version:** ...
   <!-- Add K8s-specific fields if applicable -->
   ```

4. **Show the draft to the user** and ask for confirmation or edits before filing.

5. **File the issue** using:
   ```
   gh issue create --repo ai-dynamo/dynamo --title "<title>" --body "<body>"
   ```

6. **Return the issue URL** to the user after creation.
