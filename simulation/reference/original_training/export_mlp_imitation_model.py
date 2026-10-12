#!/usr/bin/env python3

import argparse
import json
import pickle
from pathlib import Path

from baseline_eval_suite import DEFAULT_TEACHER_TRACE, ImitationMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and export a persistent mlp_imitation artifact.",
    )
    parser.add_argument(
        "--teacher-trace",
        type=str,
        default=DEFAULT_TEACHER_TRACE,
        help="Teacher trace JSON used to fit the imitation model.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="mlp_imitation_model",
        help="Output directory for the exported artifact.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for the imitation MLP.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    teacher_trace_path = Path(args.teacher_trace).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    model = ImitationMLP(
        teacher_trace_path=teacher_trace_path,
        random_state=args.random_state,
    )
    model.fit()

    payload = {
        "mode_model": model.mode_model,
        "reg_model": model.reg_model,
        "training_summary": model.training_summary,
        "teacher_trace_path": str(teacher_trace_path),
        "random_state": int(args.random_state),
    }

    artifact_path = outdir / "mlp_imitation_model.pkl"
    with open(artifact_path, "wb") as handle:
        pickle.dump(payload, handle)

    with open(outdir / "training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(model.training_summary, handle, indent=2)

    print(json.dumps({"artifact_path": str(artifact_path), **model.training_summary}, indent=2))


if __name__ == "__main__":
    main()
