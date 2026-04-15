from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def add_rows(rows: list[dict], csv_path: Path, source: str, method: str) -> None:
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        rows.append(
            {
                "source": source,
                "method": method,
                "train_dataset": row.get("train_dataset", method),
                "test_dataset": row.get("test_dataset"),
                "test_accuracy": row.get("test_accuracy"),
                "test_loss": row.get("test_loss"),
                "path": str(csv_path),
            }
        )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    results_root = repo_root / "artifacts" / "mnist_otdd"
    output_dir = results_root / "aggregate_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    add_rows(rows, results_root / "unified_benchmark_full_run1" / "method_results.csv", "unified_benchmark_full_run1", "benchmark")
    add_rows(rows, results_root / "simplecnn_scdd_full_gpu" / "student_cross_eval.csv", "simplecnn_scdd_full_gpu", "simplecnn_scdd")
    add_rows(rows, results_root / "resnet50_teacher_distill_run1" / "student_cross_eval.csv", "resnet50_teacher_distill_run1", "resnet50_teacher_distill")
    add_rows(rows, results_root / "resnet50_teacher_distill_run2" / "student_cross_eval.csv", "resnet50_teacher_distill_run2", "resnet50_teacher_distill")
    add_rows(rows, results_root / "resnet50_moe_proto_full_run1" / "student_cross_eval.csv", "resnet50_moe_proto_full_run1", "resnet50_moe_proto")

    summary_rows = []
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / "all_experiment_results.csv", index=False)
        for (source, method), group in df.groupby(["source", "method"]):
            if "test_accuracy" not in group.columns:
                continue
            acc = pd.to_numeric(group["test_accuracy"], errors="coerce").dropna()
            if acc.empty:
                continue
            summary_rows.append(
                {
                    "source": source,
                    "method": method,
                    "mean_accuracy": float(acc.mean()),
                    "min_accuracy": float(acc.min()),
                    "max_accuracy": float(acc.max()),
                    "num_rows": int(len(acc)),
                }
            )
        pd.DataFrame(summary_rows).sort_values(["mean_accuracy", "min_accuracy"], ascending=False).to_csv(
            output_dir / "all_experiment_summary.csv", index=False
        )
    else:
        pd.DataFrame(columns=["source", "method", "train_dataset", "test_dataset", "test_accuracy", "test_loss", "path"]).to_csv(
            output_dir / "all_experiment_results.csv", index=False
        )
        pd.DataFrame(columns=["source", "method", "mean_accuracy", "min_accuracy", "max_accuracy", "num_rows"]).to_csv(
            output_dir / "all_experiment_summary.csv", index=False
        )

    payload = {
        "generated_from": str(results_root),
        "num_result_rows": len(rows),
        "num_summary_rows": len(summary_rows),
    }
    (output_dir / "aggregate_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
