# Current server mission: EXP-15

Until `results/exp15/AGENT_STATUS.md` says `COMPLETE`, the first task of a
server-side coding agent is EXP-15. Read these files completely before editing
or submitting a training job:

1. `docs/EXP15_SERVER_AGENT.md` — authoritative execution order and scientific
   contract.
2. `contracts/exp15.yaml` — machine-readable arms, resources, gates, and
   required deliverables.
3. `REGISTRY.md` — experiment history and evaluation-scale restrictions.

Then run:

```bash
bash scripts/exp15/00_agent_preflight.sh
```

Continue from the first unchecked milestone in
`results/exp15/AGENT_STATUS.md`. Do not submit the 24-Worker job until the
implementation, data, native-parity, one-GPU, one-Worker, and two-Worker DDP
gates in the runbook all pass. A normal code, test, data-format, OOM, or launch
error is not a reason to stop: diagnose it, make a focused fix on
`exp15-native-orca`, rerun the failed gate, and record the evidence. Ask the
user only for an unavailable mount, permission, credential, or a scientific
choice not resolved by the runbook.

Never pull inside a GPU Pod. Pods must run a clean, fixed full commit hash and
write checkpoints, logs, configs, predictions, and provenance to shared
storage. Never use `git reset --hard`, force-push, or discard another person's
changes.
