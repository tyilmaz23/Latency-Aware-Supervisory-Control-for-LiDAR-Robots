#!/usr/bin/env python3

import csv
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest


@dataclass
class EpisodeRecord:
    controller: str
    episode: int
    seed: int
    outcome: str
    reason: str
    steps: int
    total_reward: float
    path_length_m: float
    efficiency: float
    initial_distance_m: float
    final_distance_m: float
    decision_updates: int


def bootstrap_ci(
    values: Sequence[float],
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        value = float(arr[0])
        return value, value

    rng = np.random.default_rng(random_state)
    samples = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, arr.size, size=arr.size)
        samples.append(float(np.mean(arr[indices])))
    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return lo, hi


def rate_stats(
    records: Sequence[EpisodeRecord],
    outcome_name: str,
    n_bootstrap: int = 2000,
    random_state: int = 42,
) -> Dict[str, float]:
    flags = np.asarray(
        [1.0 if record.outcome == outcome_name else 0.0 for record in records],
        dtype=np.float64,
    )
    ci_low, ci_high = bootstrap_ci(flags, n_bootstrap=n_bootstrap, random_state=random_state)
    return {
        "count": int(np.sum(flags)),
        "rate": float(np.mean(flags)),
        "std": float(np.std(flags)),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def metric_stats(
    values: Sequence[float],
    n_bootstrap: int = 2000,
    random_state: int = 42,
) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    ci_low, ci_high = bootstrap_ci(arr, n_bootstrap=n_bootstrap, random_state=random_state)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def summarize_controller(
    records: Sequence[EpisodeRecord],
    n_bootstrap: int = 2000,
    random_state: int = 42,
) -> Dict[str, object]:
    return {
        "episodes": len(records),
        "reached": rate_stats(records, "reached", n_bootstrap=n_bootstrap, random_state=random_state),
        "collision": rate_stats(records, "collision", n_bootstrap=n_bootstrap, random_state=random_state),
        "timeout": rate_stats(records, "timeout", n_bootstrap=n_bootstrap, random_state=random_state),
        "steps": metric_stats([record.steps for record in records], n_bootstrap, random_state),
        "path_length_m": metric_stats([record.path_length_m for record in records], n_bootstrap, random_state),
        "efficiency": metric_stats([record.efficiency for record in records], n_bootstrap, random_state),
        "total_reward": metric_stats([record.total_reward for record in records], n_bootstrap, random_state),
        "decision_updates": metric_stats([record.decision_updates for record in records], n_bootstrap, random_state),
    }


def exact_mcnemar(
    reference_records: Sequence[EpisodeRecord],
    candidate_records: Sequence[EpisodeRecord],
    outcome_name: str,
) -> Dict[str, float]:
    if len(reference_records) != len(candidate_records):
        raise ValueError("McNemar comparison requires equal-length paired episode lists.")

    ref_flags = np.asarray([record.outcome == outcome_name for record in reference_records], dtype=bool)
    cand_flags = np.asarray([record.outcome == outcome_name for record in candidate_records], dtype=bool)

    b = int(np.sum(ref_flags & ~cand_flags))
    c = int(np.sum(~ref_flags & cand_flags))
    discordant = b + c
    p_value = 1.0 if discordant == 0 else float(
        binomtest(min(b, c), n=discordant, p=0.5, alternative="two-sided").pvalue
    )
    return {
        "discordant_ref_yes_candidate_no": b,
        "discordant_ref_no_candidate_yes": c,
        "p_value": p_value,
    }


def load_episode_records(path: Path) -> List[EpisodeRecord]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        records = []
        for row in reader:
            records.append(
                EpisodeRecord(
                    controller=row["controller"],
                    episode=int(row["episode"]),
                    seed=int(row["seed"]),
                    outcome=row["outcome"],
                    reason=row["reason"],
                    steps=int(row["steps"]),
                    total_reward=float(row["total_reward"]),
                    path_length_m=float(row["path_length_m"]),
                    efficiency=float(row["efficiency"]),
                    initial_distance_m=float(row["initial_distance_m"]),
                    final_distance_m=float(row["final_distance_m"]),
                    decision_updates=int(row["decision_updates"]),
                )
            )
    return records


def load_json(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_records(baseline_dir: Path, actual_dir: Path) -> Dict[str, List[EpisodeRecord]]:
    records: Dict[str, List[EpisodeRecord]] = {}
    for csv_path in sorted(baseline_dir.glob("*_episodes.csv")):
        controller_name = csv_path.name.replace("_episodes.csv", "")
        records[controller_name] = load_episode_records(csv_path)

    actual_csv = actual_dir / "actual_llm_episodes.csv"
    if not actual_csv.is_file():
        raise FileNotFoundError(f"Missing actual LLM episode file: {actual_csv}")
    records["actual_llm"] = load_episode_records(actual_csv)
    return records


def verify_alignment(records_by_controller: Dict[str, List[EpisodeRecord]]) -> None:
    controller_names = list(records_by_controller)
    ref_name = controller_names[0]
    ref_records = records_by_controller[ref_name]
    ref_keys = [(record.episode, record.seed) for record in ref_records]
    ref_init = [record.initial_distance_m for record in ref_records]

    for controller_name in controller_names[1:]:
        records = records_by_controller[controller_name]
        keys = [(record.episode, record.seed) for record in records]
        if keys != ref_keys:
            raise ValueError(f"Episode/seed alignment mismatch: {ref_name} vs {controller_name}")

        init_values = [record.initial_distance_m for record in records]
        for idx, (a, b) in enumerate(zip(ref_init, init_values)):
            if abs(a - b) > 1e-9:
                raise ValueError(
                    f"Initial-distance mismatch at episode {idx}: {ref_name}={a} vs {controller_name}={b}"
                )


def build_pairwise_tests(records_by_controller: Dict[str, List[EpisodeRecord]]) -> Dict[str, object]:
    summaries = {
        name: summarize_controller(records)
        for name, records in records_by_controller.items()
    }
    pairwise: Dict[str, Dict[str, object]] = {}

    for ref_name, cand_name in combinations(records_by_controller.keys(), 2):
        ref_records = records_by_controller[ref_name]
        cand_records = records_by_controller[cand_name]

        pair_label = f"{ref_name}__vs__{cand_name}"
        pairwise[pair_label] = {
            "reference": ref_name,
            "candidate": cand_name,
            "rate_deltas_candidate_minus_reference_pp": {
                "reached": 100.0 * (
                    summaries[cand_name]["reached"]["rate"] - summaries[ref_name]["reached"]["rate"]
                ),
                "collision": 100.0 * (
                    summaries[cand_name]["collision"]["rate"] - summaries[ref_name]["collision"]["rate"]
                ),
                "timeout": 100.0 * (
                    summaries[cand_name]["timeout"]["rate"] - summaries[ref_name]["timeout"]["rate"]
                ),
            },
            "reached_mcnemar": exact_mcnemar(ref_records, cand_records, "reached"),
            "collision_mcnemar": exact_mcnemar(ref_records, cand_records, "collision"),
            "timeout_mcnemar": exact_mcnemar(ref_records, cand_records, "timeout"),
        }
    return pairwise


def build_actual_llm_focus(
    scenario_summary: Dict[str, object],
) -> List[Dict[str, object]]:
    controller_summaries = scenario_summary["controller_summaries"]
    pairwise_tests = scenario_summary["pairwise_tests"]
    focus_rows = []

    for baseline_name in ["rl_only", "gate_only", "rule_based", "mlp_imitation"]:
        pair_label = f"{baseline_name}__vs__actual_llm"
        pair = pairwise_tests[pair_label]
        focus_rows.append(
            {
                "compare_to": baseline_name,
                "actual_llm_success_rate": controller_summaries["actual_llm"]["reached"]["rate"],
                "baseline_success_rate": controller_summaries[baseline_name]["reached"]["rate"],
                "delta_success_pp": pair["rate_deltas_candidate_minus_reference_pp"]["reached"],
                "success_p_value": pair["reached_mcnemar"]["p_value"],
                "actual_llm_collision_rate": controller_summaries["actual_llm"]["collision"]["rate"],
                "baseline_collision_rate": controller_summaries[baseline_name]["collision"]["rate"],
                "delta_collision_pp": pair["rate_deltas_candidate_minus_reference_pp"]["collision"],
                "collision_p_value": pair["collision_mcnemar"]["p_value"],
                "actual_llm_timeout_rate": controller_summaries["actual_llm"]["timeout"]["rate"],
                "baseline_timeout_rate": controller_summaries[baseline_name]["timeout"]["rate"],
                "delta_timeout_pp": pair["rate_deltas_candidate_minus_reference_pp"]["timeout"],
                "timeout_p_value": pair["timeout_mcnemar"]["p_value"],
            }
        )
    return focus_rows


def merge_scenario(name: str, baseline_dir: Path, actual_dir: Path) -> Dict[str, object]:
    baseline_summary = load_json(baseline_dir / "summary.json")
    actual_summary = load_json(actual_dir / "summary.json")
    records_by_controller = collect_records(baseline_dir, actual_dir)
    verify_alignment(records_by_controller)

    controller_summaries = dict(baseline_summary["controller_summaries"])
    controller_summaries.update(actual_summary["controller_summaries"])

    controller_aux = dict(baseline_summary.get("controller_aux", {}))
    controller_aux.update(actual_summary.get("controller_aux", {}))

    merged = {
        "scenario": name,
        "baseline_dir": str(baseline_dir.resolve()),
        "actual_dir": str(actual_dir.resolve()),
        "controller_summaries": controller_summaries,
        "controller_aux": controller_aux,
        "pairwise_tests": build_pairwise_tests(records_by_controller),
    }
    merged["actual_llm_vs_baselines"] = build_actual_llm_focus(merged)
    return merged


def format_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def format_pp(value: float) -> str:
    return f"{value:+.1f} pp"


def format_p(value: float) -> str:
    if value < 1e-4:
        return "<1e-4"
    return f"{value:.4f}"


def render_report(scenarios: Dict[str, Dict[str, object]]) -> str:
    lines: List[str] = []
    lines.append("# Paired Evaluation Summary")
    lines.append("")
    for scenario_name in ["k30", "k10"]:
        scenario = scenarios[scenario_name]
        controller_summaries = scenario["controller_summaries"]
        actual_aux = scenario.get("controller_aux", {}).get("actual_llm", {})

        lines.append(f"## {scenario_name.upper()}")
        lines.append("")
        lines.append("| Controller | Success | Collision | Timeout |")
        lines.append("| --- | ---: | ---: | ---: |")
        for controller_name in ["rl_only", "gate_only", "rule_based", "mlp_imitation", "actual_llm"]:
            summary = controller_summaries[controller_name]
            lines.append(
                f"| {controller_name} | "
                f"{summary['reached']['count']}/{summary['episodes']} ({format_pct(summary['reached']['rate'])}) | "
                f"{summary['collision']['count']}/{summary['episodes']} ({format_pct(summary['collision']['rate'])}) | "
                f"{summary['timeout']['count']}/{summary['episodes']} ({format_pct(summary['timeout']['rate'])}) |"
            )
        lines.append("")

        lines.append("Actual LLM reliability:")
        lines.append(
            f"- Request failure: {actual_aux['request_failures']}/{actual_aux['total_calls']} "
            f"({format_pct(actual_aux['request_failure_rate'])})"
        )
        lines.append(
            f"- Parse failure: {actual_aux['parse_failures']}/{actual_aux['total_calls']} "
            f"({format_pct(actual_aux['parse_failure_rate'])})"
        )
        lines.append(
            f"- Latency: mean {actual_aux['latency_s']['mean']:.2f}s, "
            f"p95 {actual_aux['latency_s']['p95']:.2f}s, max {actual_aux['latency_s']['max']:.2f}s"
        )
        lines.append("")

        lines.append("| Actual LLM vs | Delta Success | p(success) | Delta Collision | p(collision) | Delta Timeout | p(timeout) |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in scenario["actual_llm_vs_baselines"]:
            lines.append(
                f"| {row['compare_to']} | "
                f"{format_pp(row['delta_success_pp'])} | {format_p(row['success_p_value'])} | "
                f"{format_pp(row['delta_collision_pp'])} | {format_p(row['collision_p_value'])} | "
                f"{format_pp(row['delta_timeout_pp'])} | {format_p(row['timeout_p_value'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    outdir = repo_root / "baseline_eval_outputs" / "paired_comparisons"
    outdir.mkdir(parents=True, exist_ok=True)

    scenarios = {
        "k30": merge_scenario(
            "k30",
            repo_root / "baseline_eval_outputs" / "k30",
            repo_root / "baseline_eval_outputs" / "k30_actual_llm_geminiflash_t15",
        ),
        "k10": merge_scenario(
            "k10",
            repo_root / "baseline_eval_outputs" / "k10",
            repo_root / "baseline_eval_outputs" / "k10_actual_llm_geminiflash_t15",
        ),
    }

    combined_summary = {"scenarios": scenarios}
    with open(outdir / "combined_summary.json", "w", encoding="utf-8") as handle:
        json.dump(combined_summary, handle, indent=2)

    report = render_report(scenarios)
    with open(outdir / "combined_report.md", "w", encoding="utf-8") as handle:
        handle.write(report)

    print(f"Wrote {outdir / 'combined_summary.json'}")
    print(f"Wrote {outdir / 'combined_report.md'}")


if __name__ == "__main__":
    main()
