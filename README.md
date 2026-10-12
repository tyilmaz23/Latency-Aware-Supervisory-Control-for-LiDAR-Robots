# Selected simulation evidence and evaluation code

This repository contains selected artifacts for the Mechatronics study of sparse supervisory modulation of a fixed navigation policy. It supports the principal simulation comparisons and reviewer questions about implementation, ablation, comparator selection, statistics, latency, and fault handling. It is a selected release, not the complete experimental workspace.

## Scope

- 189 completed configurations and all 170,300 episode records belonging to them: 134 original offline configurations, 14 gate/CBF controls, 17 live Gemini configurations, and 24 final simulation configurations.
- An additional 22,000 historical normalized-action episode records and 41,416 historical LLM query records. These belong to the earlier study, not additional configurations in the 314-run inventory.
- Fixed controller checkpoints, exact seed lists, source implementations, PPO training configuration/reward code, local-teacher labels, and recorded run configurations.
- Every committed episode's event log for all 17 live configurations and the 16 final dynamic virtual configurations, including failed and censored requests. Empty event logs contribute no lines; episode outcome records still retain those episodes.
- Full step records for the three live episodes containing the 63 reported late control ticks.

`data/all_run_inventory.json` lists all 314 completed simulation configurations (295,300 executed episodes), with a `released` flag. The other 125 configurations' episode/step records are intentionally not included. Counts are executed episodes, not independent training replications or unique scenarios. Reused seeds are paired scenarios; 500-episode stress comparisons use explicitly declared prefixes of the frozen test bank.

No physical robot trial-level records, additional hardware layouts, measured sim-to-real calibration, or hard real-time certification are supplied. The release does not make these unperformed or unavailable items available. Exploratory recurrent-model and separate backend-sensitivity appendix artifacts are outside this selected release.

## Files and provenance

- `simulation/`: unchanged experiment implementations and the model files needed for deterministic inference. Reward/PPO settings are in `trainrl.py`; APF/CBF, context and template construction in `baseline_eval_suite.py`; corrected physical-speed mapping and gate/rule behavior in `revision_suite/controllers.py`; geometry in `environment.py`; timing, cancellation and fallback in `timing.py`; actual REST worker and prompt construction in `api_worker.py` and the controller source.
- `simulation/assets/`: PPO, FSRL final/best deterministic actor tensors, distilled MLP and independent MLP. FSRL critic/optimizer checkpoints are omitted because inference uses the actor tensors directly. The PPO VecNormalize artifact is retained for provenance; observation normalization is disabled in the studied evaluator.
- `simulation/reference/original_training/`: recorded training/export source for the FSRL and local MLP comparators. The pinned simulation environment supports evaluation, not a fully packaged recreation of every historical training environment. Original FSRL training additionally requires its original FSRL/Tianshou environment; the available training history is not a multi-seed retraining study.
- `data/episodes/`: compressed JSON Lines, one object per original CSV row: `{"job_id":...,"record":{...}}`. CSV field values stay as strings to preserve stored precision and original values. All outcomes are retained for every selected run; no unsuccessful episodes have been filtered out.
- `data/run_configurations.json.gz`: original run metadata, resolved settings, software versions, source/model fingerprints, and summaries keyed by job ID. `input_hashes` also describe omitted original artifacts; absence from this subset is not a hash mismatch.
- `data/events/`: compressed JSON Lines with `job_id`, `episode`, `seed`, and the unchanged event object under `record`. These preserve contexts, decisions, request IDs, timing, activation, failure and censoring. Live nominal/fault/factorial groups are separate. Dynamic virtual K10/K30 logs use the frozen local-teacher MLP and simulation time; they are not cached Gemini replay.
- `data/normalized/`: historical normalized-action comparison and actual-Gemini robustness/reliability datasets. These are separate from the corrected physical-speed experiments. The main K10/K30 baselines and all seven additional actual-LLM robustness conditions are included.
- `analysis/archived_contrasts.json.gz`: original contrast definitions and full original multiplicity-family context. `raw_available` indicates whether both compared runs' raw records are supplied. Unavailable contrast members are context only; their results cannot be independently reconstructed from this subset.
- `MANIFEST.json`: SHA-256 and byte sizes for the release payload, excluding the manifest itself and locally generated `audit_result.json`.

Consolidation changes file organization and compression only. Episode fields and event objects are retained. For runs resumed after an interruption, only the event/step file referenced by the final committed episode record is selected; unfinished attempts are excluded. Absolute path hints may remain inside unchanged historical metadata or serialized models. No API keys or environment files are included. Third-party Tianshou provenance and its MIT notice accompany the source reference.

## Audit existing evidence without simulations or API calls

From the repository root, in an environment with Python 3.10 or later:

```text
python -m pip install -r analysis/requirements.txt
python analysis/audit_release.py
```

The script checks payload hashes, source/model fingerprints, complete episode sequences, exact seeds, outcome counts, paired scene hashes, exact McNemar tests, paired bootstrap confidence intervals (2,000 replicates; RNG seed 20260928), and Holm adjustments for complete released families. It checks query/failure accounting, activation readiness, the 92,764 live queries, 87,216 provider dispatches (85,018 provider calls confirmed in completed worker replies), 5,548 injected failures, 2,946 right-censored requests and 63 late ticks. It also checks the historical normalized datasets and counts their LLM queries.

Incomplete original multiplicity families are explicitly retained as context: Holm correction is never recalculated over only the selected contrasts. The audit recomputes individual available comparisons and verifies adjusted values only when all members of that original family are released. It writes `audit_result.json`. Historical normalized-action paired statistics use the separate original `simulation/paired_eval_report.py`; their bootstrap uses RNG seed 42 and must not be mixed with the newer evaluator's RNG seed.

Completed-response latency quantiles exclude cancelled/deadline-censored requests and must not be described as full-tail quantiles. Completion, terminal collision and geometric body contact are separate endpoints. A nonsignificant collision test does not establish equivalence or safety. Control timing is a measured diagnostic on a sampled plant, not a continuous-plant or hard real-time proof.

## Optional explicit single-job replay

Install the pinned simulation dependencies using Python 3.10, preferably in a fresh environment:

```text
python -m pip install -r simulation/requirements.txt
python tools/run_selected.py --list
python tools/run_selected.py --job native_nominal_ppo_d34d0270 --outdir replays
```

The tool uses the archived resolved configuration and frozen seed bank. Choose a fresh output directory: the selected source set intentionally changes the original complete-package fingerprint, so the original experiments must not be resumed using this release. Only one explicitly named job runs; no large queue or training starts automatically.

Live jobs require `--allow-live` and the caller's own `GEMINI_API_KEY` or `GOOGLE_API_KEY`. They incur API use and wall-clock pacing. Historical network responses and service latency are not guaranteed to recur. Recorded response/event data allow evidence auditing without any new API request. Training/export scripts are preserved references and may require adaptation of their original local paths; inference uses the released frozen models.

## Relation to Supplementary Material S1

S1 remains the methodological text submitted as supplementary material. This repository is a separate selected code/data release. Its URL must be added to the manuscript, S1 and reviewer response only after actual publication; no public URL or permanent identifier is asserted by this local folder. Repository publication strengthens the artifact-access response but does not satisfy the remaining hardware expansion or real-time certification limitations.
