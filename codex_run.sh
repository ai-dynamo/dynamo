sleep 500
while true; do
  codex exec \
    --full-auto \
    -c approval_policy="never" \
    -c sandbox_mode="workspace-write" \
    "Read AGENTS.md and then execute PLANS.md completely. Do not ask me questions. If you think you are done, first re-read PLANS.md and check for remaining work, TODOs, follow-ups, cleanup, tests, docs, or validation. Continuously compact context into PLANS.md by updating progress, decisions, remaining work, and exact next steps. Write all findings to disk. Run tests after each meaningful milestone. Make small commits as you go. If you exit, ensure PLANS.md contains a precise handoff for the next run."
  status=$?
  echo "codex exited with status $status at $(date)"
  sleep 100
done
