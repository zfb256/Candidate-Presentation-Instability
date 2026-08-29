# Candidate Presentation Instability in LLM-Based Answer Verification

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22155756.svg)](https://doi.org/10.5281/zenodo.22155756)

Code and reproducibility artifacts for:

> **Candidate Presentation Instability in Answer Verification by Large Language
> Models: A Label-Symmetric Benchmark and Controlled Study**  
> Feibao Zhuo

An LLM verifier can reverse its correctness verdict when a candidate answer is
repackaged without changing its content. This repository contains the
label-symmetric benchmark, frozen prompts, retained per-item outputs, manifests, and
analysis code used to measure that instability. The manuscript is intentionally
not included in this repository.

## Study snapshot

- 750 source groups from five public datasets
- one correct and one incorrect candidate per group
- one canonical presentation and six label-preserving transformations per candidate
- 10,500 benchmark items in total
- eight verifiers under the P1 baseline
- P2--P4 and P6 controls for Qwen2.5-3B, Qwen2.5-14B, and DeepSeek V4 Flash
- two-stage candidate withholding (P5) for Qwen2.5-14B and DeepSeek V4 Flash
- same-prompt repeat measurements for six verifiers on a 200-group subset
- five-sample self-consistency (SC@5) for Qwen2.5-14B

## Metrics

| Quantity | Meaning |
|---|---|
| **IFR** | invariant flip rate between canonical and label-preserving transformed presentations |
| **RFR** | repeat flip rate when the identical prompt is submitted twice |
| **RR** | responsiveness rate on canonical incorrect candidates |
| coverage | share of outputs satisfying the declared JSON schema |

Accuracy, incorrect-class F1, Brier score, bootstrap confidence intervals, and
invalid-output sensitivity bounds are also retained in the reports.

## Repository layout

```text
src/bav/            parsing, prompt construction, and metric code
scripts/            dataset, runner, evaluation, and audit scripts
data/processed/     10,500-item benchmark and 200-group subset
data/README.md      upstream filenames, revisions, and SHA-256 hashes
runs/               frozen prompts, per-item verifier outputs, and run manifests
reports/formal/     aggregated evaluations and sensitivity summaries
docs/               frozen design record and implementation audit
```

Every released JSONL artifact has a sibling manifest or is covered by a report
that records its inputs, hashes, model settings, and seeds.

`reports/formal/` is the shortest route from the retained outputs to the published
numbers. `evaluations.json` and `evaluations.csv` hold the per-verifier,
per-protocol metrics with their bootstrap intervals, `exact_repeat.json` holds
the same-prompt repeat control, `stability_sensitivity.json` holds the
invalid-as-non-flip and invalid-as-flip bounds, and `api_cost.json` records the
hosted request and token counts.

## Data and licensing

The upstream files under `data/raw/` are not redistributed. Download them from
their original sources and verify them against `data/README.md` before rebuilding
the benchmark. Derived benchmark items remain subject to the upstream dataset
licenses. Model outputs remain subject to the corresponding providers' terms.

## Environment

The benchmark builder, prompt exporter, evaluator, and audit use the Python 3.10+
standard library. Local inference additionally requires the versions recorded in
the manifests (PyTorch 2.5.1, CUDA 12.4, and vLLM 0.6.6.post1 for the retained
runs). API runs use Python's standard-library HTTP client.

Copy `.env.example` to `.env` and fill in only the providers you intend to run.
Never commit `.env`.

## Validate the released snapshot

```bash
python3 -B scripts/check_revision.py
```

This checks dataset structure and hashes, frozen prompt hashes, oracle behavior,
retained formal results, repeat controls, and selected regression invariants.

## Rebuild the benchmark and prompts

Place the verified upstream snapshots under `data/raw/`, then run:

```bash
python3 -B scripts/build_cross_task_dataset.py
python3 -B scripts/build_cross_task_dataset.py \
  --limit-per-dataset 40 \
  --output data/processed/cross_task_v3_probe200.jsonl

for protocol in p1_direct_front p2_direct_last p3_careful_direct_front p4_constraint_first_last; do
  python3 -B scripts/export_prompts.py --protocol "$protocol"
  python3 -B scripts/export_prompts.py \
    --items data/processed/cross_task_v3_probe200.jsonl \
    --protocol "$protocol" \
    --output "runs/probes/${protocol}_probe200.jsonl"
done

python3 -B scripts/make_oracle_outputs.py
```

## Rerun experiments

The repository already contains the retained outputs. To perform a clean rerun,
first work in a separate clone or move `runs/formal` aside; the runner refuses to
overwrite an existing formal run.

```bash
bash scripts/run_formal.sh
bash scripts/run_qwen14_p5.sh

bash scripts/run_exact_repeat.sh local
bash scripts/run_exact_repeat.sh deepseek
bash scripts/run_exact_repeat.sh doubao

python3 -B scripts/summarize_exact_repeat.py
python3 -B scripts/aggregate_evaluations.py
python3 -B scripts/summarize_stability_sensitivity.py
python3 -B scripts/summarize_api_cost.py
```

Interrupted formal runs resume only with `bash scripts/run_formal.sh --resume`
and only when the saved run identity matches the inputs and configuration.

## Reporting conventions

- Invalid outputs count as wrong for accuracy and as failures for RR.
- IFR is conditional on pairs valid at both presentations; sensitivity bounds
  treating invalid pairs as non-flips and flips are provided separately.
- Each output row holds the parsed prediction, confidence, rationale, and parse
  status; an invalid row additionally holds a truncated excerpt of the returned
  text. Full raw responses are not retained.
- Hosted calls record a length-limited finish reason as a parse failure. The
  open-weight path records no finish reason, so an open-weight response that
  reaches the generation limit but still parses counts as valid.
- P5 freezes one candidate-independent Stage-A record per source group and reuses
  it across that group's 14 candidate presentations.

## Citation

See `CITATION.cff`. The repository is archived at
[10.5281/zenodo.22155756](https://doi.org/10.5281/zenodo.22155756), which always resolves to the
latest release.

## License

Code and documentation are released under the MIT License. Dataset-derived
items and model outputs retain the additional restrictions described above.
