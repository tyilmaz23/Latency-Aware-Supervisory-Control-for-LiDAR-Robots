# Reviewer evidence map

| Question | Released evidence | Scope of the response |
|---|---|---|
| R1.1: theoretical grounding and claims | Controller/geometry implementations, all selected outcomes, separate terminal-collision/body-contact endpoints | Empirical evidence only; no safety/invariance proof |
| R1.2: fair update cadence, training budgets, checkpoint choice, classical reference | Cadence/native/checkpoint/validation outcome banks; FSRL final/best actors and training configuration; DWA validation and test resolved settings; final dynamic DWA comparison | Fixed-policy comparisons and predefined checkpoint sensitivity; custom DWA, not ROS Nav2 DWB or TEB; no multi-seed full retraining parity claim |
| R1.3: code, hyperparameters, seeds, models, traces; hardware and sim-to-real | Source, inference environment, reward/training settings, seeds, frozen models, teacher trace, selected raw episodes/events, provenance hashes | Selected public simulation artifacts once actually uploaded. Physical raw trials, extra hardware layouts and measured sim-to-real calibration remain unavailable/unperformed |
| R1.4: gate/scaffold/content contribution, attribution and statistics | Full selected ablation/mapping families, gate controls, 17 live configurations including factorial controls; original contrasts and independent audit | Distinguishes decision source, rules, gate and mapping; completion gain cannot uniquely be assigned to LLM reasoning |
| R1.5: delayed-only activation, timing tails, failure/fallback and real-time | Actual REST worker and supervisor source, all committed live events, original reliability query traces, dynamic virtual events, all three overrun step logs | Checks activation readiness and retains censoring. Worker scheduling and sampled-plant results do not establish hard real-time guarantees |
| R2.1: rule/emergency effects | Rule and emergency configurations, positive and negative outcomes, mapping diagnostics | No favorable-outcome filtering; body contact and terminal outcomes separately retained |
| R2.2: PPO reward weights and hyperparameters | `simulation/trainrl.py`, exact model and normalization artifact, stored configuration/version metadata | Reward coefficients and training implementation exposed |
| R2.3: CBF-QP/lookahead/body/discrete-time limits | Original CBF implementation, 14 additional gate/CBF controls, native outcomes and body-contact fields | Radius/lookahead effects are empirical, not body/braking/intersample guarantees |
| R2.4: APF equations and force-to-command mapping | `simulation/baseline_eval_suite.py`, native APF configurations and outcomes | Actual implementation is available for inspection and deterministic replay |
| R2.5: gate-only definition and modulation | `revision_suite/controllers.py`, gate controls, full ablation/mapping outcome banks | Fixed no-op supervisor tuple and shared rules are inspectable; gate alone is not presented as content modulation |

## Selection rule

The released original offline jobs are those carrying at least one of the groups `native`, `ablation`, `mapping`, `cadence`, `checkpoint`, `validation`; group membership and aliases are retained in their archived job metadata. All 14 controls, 17 live and 24 final configurations are included. This preserves full condition families rather than selecting successful runs. It excludes original `repair`, `virtual`, `geometry`, `domain`, and `sensitivity` jobs unless deduplicated job metadata also belongs to a selected group. Original virtual/geometry controls are complemented by the released final dynamic virtual and additional CBF controls; omitted groups are not claimed to be independently reconstructed.

All 314 jobs' counts and release flags are in the inventory. Of 1,584 archived newer-evaluator contrast records, 906 have both raw inputs in this subset. The audit verifies exact raw-input comparisons and fully available multiplicity families; unavailable members remain identifiable original context. Historical normalized-action data are a separate bank and never pooled with the new experiments.

This map explains the purpose of each selected artifact. It does not assert that every reviewer request has been fully satisfied.
