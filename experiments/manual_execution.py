"""Operações manuais e proteção de artefatos para o menu do terminal."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np

from experiments import (evaluate_false_positives, evaluate_feature_importance, evaluate_multiscale_context as e01,
                         evaluate_regions, evaluate_slic_oracle, evaluate_thresholds)
from experiments.inspect_dataset import inspect_images
from experiments.reporting import REPORT_FILES, metric_rows, metrics_table, read_report, render_summary, write_summary
from experiments.train_regions import build_datasets, train_from_datasets
from src.dataset.classes import DamageClass
from src.feature_extractor import DEFAULT_FEATURE_GROUPS
from src.region_dataset import PIPELINE_CONFIG
from src.region_features import FEATURE_NAMES
from src.region_prediction import model_feature_groups


DATASET = Path("data/BFDD")
BASELINE = Path("results/region_full")
DIAGNOSTICS = Path("results/diagnostics")
OPTIMIZATION = Path("results/optimization/E01_1_multiscale_context")
MANUAL = Path("results/manual")


@dataclass(frozen=True)
class Diagnostic:
    code: str
    title: str
    folder: str
    dependencies: tuple[tuple[str, tuple[str, ...]], ...]
    run: object
    baseline_files: tuple[str, ...] = ()


D01 = "D01_slic_oracle"
D02 = "D02_false_positives"
D03 = "D03_threshold_analysis"
D04 = "D04_feature_importance"
D02_INPUT = (D02, ("configuration.json", "region_statistics.csv"))
D03_INPUT = (D03, ("optimal_thresholds.json", "per_image_comparison.csv"))
EXPERIMENTS = (
    Diagnostic("D01", "Limite teórico do SLIC", D01, (), evaluate_slic_oracle.run_experiment,
               ("random_forest.joblib", "pixel_validation_metrics.json", "pixel_validation_per_image.json")),
    Diagnostic("D02", "Falsos positivos", D02, ((D01, ("configuration.json", "region_statistics.csv")),),
               evaluate_false_positives.run_experiment, ("random_forest.joblib", "pixel_validation_per_image.json")),
    Diagnostic("D03", "Limiares de decisão", D03, (D02_INPUT,), evaluate_thresholds.run_experiment,
               ("pixel_validation_per_image.json",)),
    Diagnostic("D04", "Importância das features", D04, (D02_INPUT, D03_INPUT),
               evaluate_feature_importance.run_experiment, ("random_forest.joblib", "pixel_validation_per_image.json")),
)


def safe_run_path(namespace: str, name: str) -> Path:
    """Aceita um nome simples em um espaço exclusivo de execução manual."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,59}", name):
        raise ValueError("Use até 60 letras ASCII, números, hífens ou sublinhados no nome da execução.")
    path = MANUAL / namespace / name
    if path.resolve() != MANUAL.resolve() / namespace / name or MANUAL.is_symlink():
        raise ValueError("O destino redireciona para fora da pasta da execução.")
    return path


def allow_write(root: Path, targets: tuple[str, ...], ask=input) -> bool:
    """Solicita SIM para regravar arquivos da execução manual escolhida."""
    try:
        relative = root.absolute().relative_to(MANUAL.absolute())
    except ValueError as error:
        raise ValueError(f"Saída fora da árvore manual: {root}") from error
    if MANUAL.is_symlink() or root.resolve() != MANUAL.resolve() / relative:
        raise ValueError(f"Saída fora da árvore manual ou redirecionada: {root}")
    for target in targets:
        path = root / target
        if path.resolve() != root.resolve() / target or (
                path.is_dir() and any(child.is_symlink() or not child.resolve().is_relative_to(root.resolve())
                                      for child in path.rglob("*"))):
            raise ValueError(f"Saída redirecionada recusada: {path}")
    existing = [target for target in targets if (root / target).exists()]
    if not existing:
        return True
    print(f"Artefatos existentes em {root}: {', '.join(existing)}")
    return ask("Digite SIM para substituir os artefatos desta etapa: ").strip() == "SIM"


def require_files(root: Path, *names: str) -> None:
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Faltam artefatos em {root}: {', '.join(missing)}.")


def manifest() -> dict:
    data = read_report(BASELINE / "split.json")
    train, validation = data["train"], data["validation"]
    if (len(train) != 539 or len(validation) != 132 or len(set(train + validation)) != 671
            or set(train + validation).intersection(data.get("test_reserved", ()))):
        raise ValueError("O manifesto deve conter os 539 IDs de treino e os 132 de validação distintos.")
    return data


def baseline_config() -> dict:
    model = joblib.load(BASELINE / "random_forest.joblib")
    config = model["pipeline_config"]
    if (tuple(model["feature_names"]) != FEATURE_NAMES or config["labeling"]["class_id"] != DamageClass.STAIN
            or model["image_ids"] != manifest()["train"] or model["model"].n_features_in_ != len(FEATURE_NAMES)):
        raise ValueError("A configuração do modelo original é incompatível com o manifesto ou com Stain.")
    return config


def validate_datasets(source: Path, *, experimental: bool = False) -> None:
    split = manifest()
    config = baseline_config()
    for name in ("train", "validation"):
        with np.load(source / f"{name}.npz", allow_pickle=False) as archive:
            names = tuple(archive["feature_names"])
            if ((experimental and names != (*FEATURE_NAMES, *e01.MULTISCALE_NAMES))
                    or archive["image_ids"].tolist() != split[name]
                    or json.loads(archive["pipeline_config"].item()) != config):
                raise ValueError(f"Configuração, divisão ou colunas incompatíveis: {source / (name + '.npz')}")


def inspect(subset: str, size: int) -> None:
    inspect_images(DATASET, manifest()[subset][:size])


def build_baseline_copy(output: Path, feature_groups=DEFAULT_FEATURE_GROUPS) -> None:
    config = baseline_config()
    if config != PIPELINE_CONFIG:
        raise ValueError("Os padrões do construtor diferem da configuração registrada na baseline.")
    build_datasets(DATASET, output, False, manifest_path=BASELINE / "split.json", feature_groups=feature_groups)


def train_baseline_copy(output: Path, source: Path) -> None:
    validate_datasets(source)
    destination_split = output / "split.json"
    if destination_split.is_file() and read_report(destination_split) != manifest():
        raise ValueError("O manifesto do destino difere da divisão registrada.")
    train_from_datasets(source, output, manifest_path=BASELINE / "split.json")


def evaluate_saved_model(output: Path, model_path: Path, image_id: str | None = None, *,
                         threshold: float | None = None, optimize: bool = False) -> None:
    ids = manifest()["validation"]
    if image_id is not None and image_id not in ids:
        raise ValueError("O ID deve pertencer à validação registrada.")
    artifact = joblib.load(model_path)
    model_feature_groups(artifact)
    if artifact["pipeline_config"] != baseline_config() or artifact["image_ids"] != manifest()["train"]:
        raise ValueError("Configuração ou IDs de treino do modelo incompatíveis com o manifesto de Stain.")
    if image_id is not None and optimize:
        raise ValueError("A busca de limiar ótimo está disponível na validação.")
    output.mkdir(parents=True, exist_ok=True)
    if image_id is None:
        evaluate_regions.evaluate_validation(
            artifact, DATASET, ids, output, threshold=threshold, optimize=optimize, model_path=model_path,
        )
        return
    started = perf_counter()
    record, panels = evaluate_regions.evaluate_image(artifact, DATASET, image_id, threshold=threshold)
    evaluate_regions.visualize_prediction(*panels, output / "prediction.png")
    e01.write_json(output / "global_metrics.json", {
        "images": 1, **record, **evaluate_regions.stain_iou_summary([record]),
        "configuration": evaluate_regions.evaluation_configuration(artifact, model_path, threshold),
        "processing_seconds": perf_counter() - started,
    })


def diagnostic_paths(folder: str) -> list[Path]:
    return [DIAGNOSTICS / folder, *sorted((MANUAL / folder).glob("*"))]


def report_paths() -> list[Path]:
    candidates = [BASELINE, OPTIMIZATION, *(DIAGNOSTICS / item.folder for item in EXPERIMENTS)]
    if MANUAL.is_dir():
        candidates.extend(sorted({path.parent for filename in REPORT_FILES for path in MANUAL.rglob(filename)}))
    return list(dict.fromkeys(path for path in candidates if any((path / name).is_file() for name in REPORT_FILES)))


def compare_reports(paths: list[Path]) -> None:
    rows = []
    for root in paths:
        for filename in ("pixel_validation_metrics.json", "global_metrics.json", "global_statistics.json",
                         "validation_metrics.json", "individual_metrics.json"):
            path = root / filename
            if path.is_file():
                rows.extend((f"{root.as_posix()} · {label}", values) for label, values in metric_rows(read_report(path)))
    print("\n".join(metrics_table(rows)) if rows else "Nenhuma métrica comparável encontrada.")


def process(title: str, output: Path, operation, *, summarize: bool = True) -> None:
    print(f"\n{title}\nDestino: {output}\nEtapa iniciada.", flush=True)
    started = perf_counter()
    operation()
    if summarize:
        write_summary(output)
    print(f"Concluído em {perf_counter() - started:.2f} s · {output}")
    if summarize:
        print(render_summary(output))
