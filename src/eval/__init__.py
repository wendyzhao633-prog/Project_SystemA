__all__ = ["export_baseline_report_artifacts"]


def __getattr__(name: str):
    if name == "export_baseline_report_artifacts":
        from src.eval.report_artifacts import export_baseline_report_artifacts

        return export_baseline_report_artifacts
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
