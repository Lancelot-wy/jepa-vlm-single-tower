Execute the current EXP-15 server mission.

Read `CLAUDE.md`, `docs/EXP15_SERVER_AGENT.md`, `contracts/exp15.yaml`, and
`results/exp15/AGENT_STATUS.md` completely. Run
`bash scripts/exp15/00_agent_preflight.sh`, continue from the first unchecked
milestone, implement and validate every required artifact, and submit the
24-Worker job only after all hard gates pass. Diagnose ordinary failures and
resume instead of stopping. Keep the status file and shared-storage evidence
current, commit focused fixes to `exp15-native-orca`, and never pull from a GPU
Pod.
