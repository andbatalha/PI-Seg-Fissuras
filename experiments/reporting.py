"""Apresentação quantitativa dos relatórios de execução e comparação."""

import csv
import json
from pathlib import Path


METRICS = ("Accuracy", "Precision", "Recall", "IoU", "F1", "TP", "TN", "FP", "FN")
HEADERS = ("Accuracy", "Precision", "Recall", "IoU", "F1", "TP", "TN", "FP", "FN",
           "Limiar", "IoU média", "IoU mediana", "Δ IoU vs. B00", "s/imagem")
REPORT_FILES = (
    "dataset_statistics.json", "validation_metrics.json", "pixel_validation_metrics.json",
    "global_metrics.json", "global_statistics.json", "training_statistics.json",
    "false_positives_by_presence.json", "false_positives_by_oracle.json",
    "preparation_statistics.json", "individual_metrics.json",
)


def read_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def saved_model_paths(baseline: Path, results: Path) -> list[Path]:
    """Lista modelos do projeto usando os nomes adotados pelos executores."""
    names = {"random_forest.joblib", "E01.1-A.joblib", "E01.1-B.joblib"}
    candidates = (baseline / "random_forest.joblib",
                  *(path for path in sorted(results.rglob("*.joblib")) if path.name in names))
    return [path for path in dict.fromkeys(candidates) if path.is_file()]


def metric_rows(report: dict) -> list[tuple[str, dict]]:
    """Extrai linhas conhecidas, preservando o escopo de cada métrica."""
    rows = []
    if isinstance(report.get("comparison"), list):
        rows.extend((f"{row['variant']} · {row['policy']}", row) for row in report["comparison"])
    if isinstance(report.get("metrics"), dict):
        rows.append(("Avaliação global por pixel", {
            **report.get("counts", {}), **report["metrics"],
            "threshold": report.get("configuration", {}).get("threshold"),
            "mean_iou_with_stain": report.get("mean_iou_with_stain"),
            "median_iou_with_stain": report.get("median_iou_with_stain"),
            "mean_inference_seconds": report.get("seconds_per_image", report.get("inference_seconds")),
        }))
    if "confusion_matrix_rows_true_columns_predicted_0_1" in report and "accuracy" in report:
        (tn, fp), (fn, tp) = report["confusion_matrix_rows_true_columns_predicted_0_1"]
        rows.append(("Validação por região", {**{key: report.get(key.lower()) for key in METRICS[:5]},
                                              "TP": tp, "TN": tn, "FP": fp, "FN": fn}))
    subgroups = {"baseline": "Baseline", "majority": "Oracle Majority", "max_iou": "Oracle Max-IoU",
                 "default": "Baseline padrão", "d03": "Limiar D03"}
    for key, label in subgroups.items():
        row = report.get(key)
        if isinstance(row, dict):
            values = {**row, **row.get("metrics", {}), **row.get("counts", {})}
            if any(metric in values for metric in METRICS):
                if key == "max_iou":
                    values["threshold"] = report.get("max_iou_occupancy_cutoff")
                rows.append((label, values))
    for field in ("optima", "references"):
        for key, row in report.get(field, {}).items():
            if isinstance(row, dict):
                rows.append((subgroups.get(key, key), row))
    return rows


def format_value(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", r"\|")


def metrics_table(rows: list[tuple[str, dict]]) -> list[str]:
    if not rows:
        return []
    lines = ["| Avaliação | " + " | ".join(HEADERS) + " |",
             "| " + " | ".join(["---"] * (len(HEADERS) + 1)) + " |"]
    for name, row in rows:
        values = [f"{row[key]:.2%}" if isinstance(row.get(key), (int, float)) else "—" for key in METRICS[:5]]
        values.extend(format_value(row.get(key)) for key in (*METRICS[5:], "threshold"))
        for key in ("mean_iou_with_stain", "median_iou_with_stain", "iou_delta_vs_b00"):
            values.append(f"{row[key]:.2%}" if row.get(key) is not None else "—")
        values.append(format_value(row.get("mean_inference_seconds")))
        lines.append("| " + " | ".join([name, *values]) + " |")
    return lines


def quantitative_details(report: dict) -> list[str]:
    """Resume indicadores escalares e agrupamentos pequenos de interesse."""
    keys = ("training", "validation", "with_stain", "without_stain", "also_selected", "not_selected",
            "B00", "E01.1-A", "E01.1-B", "occupancy", "high_probability_zero_occupancy", "changes",
            "baseline", "majority", "max_iou", "reproduction")
    details = {"Global": report, **{key: report[key] for key in keys if isinstance(report.get(key), dict)}}
    rows = [(name, key, value) for name, values in details.items() for key, value in values.items()
            if isinstance(value, (int, float)) and key not in METRICS]
    lines = ["| Grupo | Indicador | Valor |", "| --- | --- | --- |"] if rows else []
    lines.extend(f"| {name} | {key} | {format_value(value)} |" for name, key, value in rows)
    return lines


def render_summary(root: Path) -> str:
    lines = [f"# Resultados · {root.name}"]
    configuration = root / "configuration.json"
    if configuration.is_file():
        config = read_report(configuration)
        pipeline = config.get("pipeline_config", {})
        settings = {"SLIC": pipeline.get("slic", config.get("slic", config.get("slic_for_examples_only"))),
                    "Rotulagem": pipeline.get("labeling"), "Random Forest": config.get("forest_config"),
                    "Limiar fixo": config.get("fixed_threshold", config.get("d03_threshold"))}
        visible = [(key, value) for key, value in settings.items() if value is not None]
        if visible:
            lines.extend(("", "## Configuração", "", "| Parâmetro | Valor |", "| --- | --- |"))
            lines.extend(f"| {key} | {format_value(value)} |" for key, value in visible)
    for filename in REPORT_FILES:
        path = root / filename
        if path.is_file():
            report = read_report(path)
            lines.extend(("", f"## {filename}", ""))
            settings = report.get("configuration", {})
            if "feature_groups" in settings:
                groups = ", ".join(settings["feature_groups"])
                lines.append(f"- Características ({settings['feature_count']}) · {groups}")
            if "model" in settings:
                lines.append(f"- Modelo · {settings['model']}")
            if "decision" in settings:
                decisions = {"default": "Decisão padrão do estimador", "threshold": "Limiar personalizado",
                             "max_iou_validation": "Limiar que maximiza a IoU neste conjunto de validação"}
                decision = decisions.get(settings["decision"], settings["decision"])
                if settings["decision"] == "max_iou_validation" and settings["threshold"] is None:
                    decision += " · todas as regiões negativas" if report.get("images") else " · avaliação incompleta"
                lines.append(f"- {decision}")
            lines.extend(("", *metrics_table(metric_rows(report))))
            details = quantitative_details(report)
            if details:
                lines.extend(("", *details))
    for filename in ("permutation_importance.csv", "group_permutation_importance.csv"):
        path = root / filename
        if path.is_file():
            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            lines.extend(("", f"## {filename}", "", "| Corte | Característica ou grupo | Queda IoU |",
                          "| --- | --- | --- |"))
            lines.extend(f"| {row['cut']} | {row['name']} | {float(row['mean_IoU_drop']):.2%} |" for row in rows)
    return "\n".join(lines) + "\n"


def write_summary(root: Path) -> Path:
    path = root / "summary.md"
    path.write_text(render_summary(root), encoding="utf-8")
    return path


def guard_cli_output(root: Path, targets: tuple[str, ...], *, owner: str | None = None) -> None:
    """Protege artefatos originais e pede confirmação para outras saídas existentes."""
    if root.is_symlink() or any((root / target).resolve() != root.resolve() / target for target in targets):
        raise SystemExit("O destino de escrita contém um redirecionamento.")
    for area in ("diagnostics", "optimization"):
        parent = Path("results", area).resolve()
        if root.resolve().is_relative_to(parent):
            parts = root.resolve().relative_to(parent).parts
            if not parts or parts[0] != owner:
                raise SystemExit("O destino pertence a outro experimento. Use outro --output.")
    existing = [target for target in targets if (root / target).exists()]
    if not existing:
        return
    if root.resolve().is_relative_to(Path("results/region_full").resolve()):
        raise SystemExit("Saída original da baseline protegida. Use outro --output ou o menu manual.")
    try:
        confirmed = input(f"Artefatos existentes em {root}: {', '.join(existing)}. Digite SIM para substituir: ")
    except EOFError:
        confirmed = ""
    if confirmed.strip() != "SIM":
        raise SystemExit("Operação cancelada. Artefatos preservados.")
