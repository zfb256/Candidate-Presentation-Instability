# v3 implementation audit

Audit date: 2026-08-24

## Outcome

The implemented benchmark, four frozen prompt protocols, local/API runners,
two-stage API runner, parser, evaluator, and bootstrap code pass the current
regression audit. No known blocking correctness defect remains in this
implemented scope. Former model outputs were removed and are not evidence.

## Correctness defects fixed

- Pinned source hashes and row counts; rebuilt MATH from the official test split.
- Paired each invariant only with the same-polarity original.
- Enforced strict, non-empty JSON output while preserving bare LaTeX commands.
- Counted invalid and missing incorrect originals as responsiveness failures.
- Preserved duplicate group-bootstrap draws as distinct sampled clusters.
- Bound resume to inputs, model, decoding, endpoint, and runner-code hashes.
- Reused one candidate-hidden P5 Stage A per source group.
- Calibrated P6 from valid P4 rationales and recorded achieved token counts.
- Retained API attempts, thinking mode, usage, and transport errors.
- Added a six-verifier exact-repeat control on the stratified 200-group probe.
- Reused the audited two-stage path for Qwen2.5-14B through a local vLLM endpoint.

## Regression evidence

- Dataset: 750 groups, 10,500 items, 5 datasets, 2 balanced polarities, and 6
  invariant families plus the original.
- Full dataset SHA-256:
  `73db47b2bbe4ee3c65710aab0b0eee0a17c42f349abf52560fef04d9907304de`.
- P1--P4 prompt SHA-256 values:
  `659fc919208632e07475fced4f56d3ee5b04506805b9d80a11bb9fe856138c75`,
  `e56ff57fffd54e3af209ef3914d983c68da48f984e8e15d97d70e256cff57e4d`,
  `7da5fff8f30e311faf435b8910bf0f6bc944779e62358ebcd2b352506bdc6f56`,
  and `14ad6d777a0eed069f474ed342706f46d860d030b1772bddf23b2b8fd7d55d97`.
- Independent rebuilds reproduced the dataset and P1 hash exactly.
- Oracle evaluation gives accuracy/F1/responsiveness 1 and Brier/IFR 0.
- All active Python files parse, incompatible resumes and accidental overwrites
  are rejected, and the aggregate self-check passes.

## Formal-run state

The rerun formal outputs, evaluation reports, hashes, and manifests are retained
under `runs/formal/`. The runner requires a clean output path by default;
`--resume` is reserved for an interrupted run with matching identity. Both P5
runs used the audited two-stage path, the exact-repeat controls retained their
own manifests, and SC@5 used the compute-based reference path.
