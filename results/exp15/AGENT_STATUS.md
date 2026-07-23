# EXP-15 server Agent status

Status: **HANDOFF_READY — implementation and launch have not started**

The authoritative instructions are `docs/EXP15_SERVER_AGENT.md`; the
machine-readable contract is `contracts/exp15.yaml`. The server Agent must
replace the status above as milestones are completed and include command,
timestamp, commit, output path, and pass/fail evidence for every checked item.

## Milestones

- [ ] S0: clean `exp15-native-orca` checkout and server preflight
- [ ] S1: recover and verify raw EXP-13/14 provenance
- [ ] S2: implement native-Qwen train/eval parity and answer-only CE
- [ ] S3: build/audit clean VQA, Observation, and Vript Event manifests
- [ ] S4: implement faithful isolated-frame Observation/Event objectives
- [ ] S5: pass unit, one-GPU, four-GPU, two-Worker, and resume gates
- [ ] S6: materialize six configs and validate the 24-Worker topology
- [ ] S7: submit one fixed-commit 24-Worker job and record its job ID
- [ ] S8: monitor checkpoints/logs; debug and resume any failed arm
- [ ] S9: run native MVBench/TempCompass evaluation and paired statistics
- [ ] S10: commit result artifacts, write verdict, and confirm all Workers exited

## Active failure

None recorded. Do not use this line to park an ordinary implementation or
runtime error: write the root cause and the corrective retry under the relevant
milestone and continue.
