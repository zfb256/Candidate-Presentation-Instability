# Applied Intelligence revision design (frozen before v3 model runs)

Date frozen: 2026-08-23

## Governing decision

This document is the authoritative execution plan. It retains the original
five-stage structure, label-symmetric benchmark, six perturbation families,
layered runs, metric set, and journal positioning, with feasibility corrections:

- use the six available general-purpose local checkpoints plus Doubao and DeepSeek API, rather
  than unavailable local models or a redundant qwen-plus run;
- keep the API bill below RMB 100, with a measured token-cost gate before full
  runs;
- use a clean sequence of prompt controls whose pairwise contrasts change one
  declared factor at a time;
- use final-answer candidates for MATH because a generic string rewrite cannot
  certify that a fabricated reasoning chain remains mathematically consistent.

## Pre-writing brief

- Article type: empirical evaluation and reliability-measurement paper.
- Target: *Applied Intelligence*.
- Central claim: item-level accuracy is insufficient for LLM answer verifiers;
  a label-symmetric grouped benchmark exposes candidate-presentation
  instability, and mitigation effects must be interpreted jointly with
  responsiveness and cost.
- Contribution: (1) an operational definition and paired IFR metric for
  single-candidate answer verification; (2) a deterministic, label-symmetric
  benchmark with six perturbation families; (3) a factorial analysis separating
  candidate position, constraint-first instruction, and genuine candidate
  withholding; and (4) deployment guidance based on accuracy, IFR,
  responsiveness, and call cost.
- Claim boundary: the benchmark measures the six declared perturbation families
  in answer verification. It does not establish robustness for open-ended
  scoring, all semantic paraphrases, or adversarially optimized attacks.

## Benchmark design

- Source datasets: the pinned MATH test split, BoolQ validation, OpenBookQA
  test, ARC-Challenge test, and CommonsenseQA validation.
- Sampling: 150 source examples per dataset with a fixed seed; 750 groups.
- Group structure: two polarities (correct and incorrect), each with one
  original and six invariant presentation variants; 14 items per group and
  10,500 items total.
- Perturbation families: format, verbosity, paraphrase, confidence cue,
  authority cue, and irrelevant background.
- Pairing: each variant is paired only with the original of the same polarity.
  Group-level bootstrap resamples all 14 items together.
- MATH uses final-answer verification only. The rejected draft's full-solution
  candidates could pair a wrong final answer with otherwise correct reasoning;
  we remove that construct-validity confound instead of automatically
  fabricating propagated derivations that cannot be certified in general.
- Construction validation: deterministic extraction checks answer preservation,
  cross-polarity answer change, group-cell completeness, valid option labels,
  distractor leakage, strict label balance, and oracle pipeline consistency.
  Oracle consistency is a pipeline check, not semantic validation.
- No human annotation or redundant LLM equivalence audit is planned. The six
  transformations are validated deterministically; semantic equivalence beyond
  those checks remains an explicit construct-validity limitation.

## Protocol design

The specification's six-protocol structure is retained with exact contrasts:

| ID | Protocol | Primary contrast |
|---|---|---|
| P1 | Direct-front | baseline |
| P2 | Direct-last | P2 - P1: candidate-position effect |
| P3 | Careful-direct-front | P3 - P1: caution effect |
| P4 | Constraint-first-last | P4 - P2: constraint instruction effect |
| P5 | Two-stage withholding | P5 - P4: genuine withholding increment |
| P6 | Length-matched direct | P6 - P1: approximate reasoning-budget effect |

P5 uses two calls: Stage A derives an expected answer without the candidate;
Stage B receives the task, frozen Stage-A output, and candidate. It is the only
protocol described as genuine candidate withholding. P6 remains an approximate
control and will be interpreted using achieved rationale length, not its prompt
instruction alone.

For benchmark execution, the deterministic Stage-A result is computed once per
source group and reused for its 14 candidates. This prevents API nondeterminism
in Stage A from contaminating within-group IFR; deployment cost is reported both
with this cache and under the conservative two-calls-per-candidate assumption.

P1--P6 use the same 512-token generation limit and deterministic decoding;
P5 differs in its declared two-call structure rather than a larger per-call
budget. The paper will report prompt-factor effects without inferring an
internal cognitive mechanism. Self-consistency at five samples is a
compute-based reference on a representative local model.

## Metrics fixed before model runs

- Accuracy, incorrect-class F1, Brier score, and invalid-output rate.
- IFR overall, by candidate polarity, and by perturbation family.
- Responsiveness rate: proportion of incorrect-original answer-flip items
  correctly rejected.
- Paired group-bootstrap 95% confidence intervals for protocol contrasts.
- Calls, input/output tokens when exposed by the backend, latency, and estimated
  monetary cost for API runs.

No post-hoc metric or subgroup will enter the main paper without being labelled
exploratory.

The public source tasks may have appeared in model training data. The primary
protocol contrasts therefore remain within-model and item-paired; P1 differences
between providers are descriptive and are not interpreted as contamination-free
capability rankings.

## Model plan

The local verifiers are restricted to weights already available under
`/root/models`; the unavailable 27B model will not enter the v3 study:

- Qwen2.5-1.5B-Instruct
- Qwen2.5-3B-Instruct
- Qwen2.5-7B-Instruct
- Qwen2.5-14B-Instruct
- DeepSeek-R1-Distill-Qwen-7B
- InternLM2.5-7B-Chat

All six models receive P1. P2, P3, P4, and P6 are run on Qwen2.5-3B and
Qwen2.5-14B to cover ability and scale differences.
Self-consistency is run on Qwen2.5-14B. P5 uses the same OpenAI-compatible
two-stage runner for DeepSeek and for a local Qwen2.5-14B vLLM endpoint, avoiding
a second implementation whose only difference would be the transport backend.

Local sampling uses seed 20260823, recorded in each manifest. Greedy local
outputs are parsed once; malformed outputs remain invalid rather than being
regenerated with the same deterministic configuration.

The manuscript will describe this as six local verifiers, including a Qwen
scale series, an InternLM checkpoint, and one reasoning-distilled variant.

API budget is capped at RMB 100 and used for provider diversity rather than a
fifth Qwen checkpoint:

- the configured Volcengine/Ark Doubao endpoint;
- DeepSeek V4 Flash with thinking explicitly disabled.

Qwen-plus is excluded because four local Qwen checkpoints already establish the
Qwen-family pattern. Doubao receives P1 for provider diversity. DeepSeek receives
P1--P4, P6, and the two-stage P5 mechanism run. The reviewer-driven 200-group
exact-repeat control covers all six usable verifiers; the two added API runs keep
total recorded API cost below RMB 100. API self-consistency is excluded.

The Doubao Seed 2.1 Pro snapshot and DeepSeek V4 Flash are both called with
thinking explicitly disabled so protocol contrasts are not confounded by a
provider-default reasoning mode.

Formal outputs must start from an absent `runs/formal` directory. Resume is an
explicit recovery action and is accepted only when the saved run identity
matches the current inputs and decoding configuration.

## Target manuscript architecture

1. Introduction
2. Related work
3. Problem formulation and measurement framework
4. Label-symmetric benchmark and validation
5. Verification protocols and experimental setup
6. Results
7. Discussion and practical guidance
8. Conclusion

Main displays are limited to the benchmark schematic, an overall performance
table, a factorial-contrast figure, and a perturbation-family heat map. Calls,
tokens, and API cost are reported as exact values rather than collapsed onto a
single cost axis. Secondary dataset-level results and exact prompts belong in
supplementary material.
