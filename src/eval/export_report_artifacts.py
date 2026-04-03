from __future__ import annotations

import argparse
from pathlib import Path

from src.eval.report_artifacts import export_baseline_report_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Export report-ready artifacts for existing fusion runs.")
    parser.add_argument("--late-run-name", default="ecapa_video_reuse_baseline")
    parser.add_argument(
        "--early-run-names",
        nargs="+",
        default=["early_fusion_3cnn", "early_fusion_bilinear", "early_fusion_xattn"],
    )
    parser.add_argument("--report-name", default="baseline_diagnostic")
    args = parser.parse_args()

    audit = export_baseline_report_artifacts(
        repo_root=Path.cwd().resolve(),
        late_run_name=args.late_run_name,
        early_run_names=tuple(args.early_run_names),
        report_name=args.report_name,
    )
    print(f"report: {Path.cwd().resolve() / 'outputs/reports' / args.report_name / 'audit_summary.json'}")
    print(audit["longer_training_recommendation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
