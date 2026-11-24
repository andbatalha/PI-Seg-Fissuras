"""Carregamento e preparação dos dados exibidos pelo inspetor Streamlit."""

import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import streamlit as st
from skimage.segmentation import find_boundaries

from experiments.evaluate_regions import count_binary_pixels, pixel_metrics
from src.dataset.classes import DamageClass
from src.dataset.loader import load_image, load_label
from src.dataset.masks import create_binary_mask
from src.region_prediction import predict_regions


ERROR_COLORS = {
    "TP": (0, 200, 83),
    "FP": (239, 45, 54),
    "FN": (41, 105, 255),
}


def file_signature(path: Path) -> tuple[int, int] | None:
    """Identifica alterações no arquivo para invalidar o cache correspondente."""
    if not path.is_file():
        return None
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


@st.cache_data(show_spinner=False)
def load_catalog(results: Path, signatures: tuple) -> tuple[dict, dict, dict]:
    """Lê os IDs permitidos, a presença de Stain e as métricas já registradas."""
    manifest = json.loads((results / "split.json").read_text(encoding="utf-8"))
    subsets = {name: manifest[name] for name in ("train", "validation")}
    report_path = results / "pixel_validation_per_image.json"
    validation_ids = set(subsets["validation"])
    report = {}
    if report_path.is_file():
        rows = json.loads(report_path.read_text(encoding="utf-8"))
        report = {row["image_id"]: row for row in rows if row["image_id"] in validation_ids}

    presence = {}
    for subset, allowed_ids in subsets.items():
        path = results / f"{subset}.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as data:
                ids = data["image_ids"].tolist()
                has_stain = np.zeros(len(ids), dtype=bool)
                # Cada índice pode repetir em várias linhas, uma para cada região.
                np.logical_or.at(has_stain, data["image_indices"], data["target_counts"] > 0)
            allowed = set(allowed_ids)
            presence.update((name, bool(found)) for name, found in zip(ids, has_stain) if name in allowed)
    presence.update((name, row["has_stain"]) for name, row in report.items())
    return subsets, presence, report


@st.cache_resource(show_spinner=False)
def load_model(path: Path, signature: tuple[int, int]) -> dict:
    """Reabre o artefato confiável do projeto e confere a classe treinada."""
    artifact = joblib.load(path)
    if artifact["pipeline_config"]["labeling"]["class_id"] != DamageClass.STAIN:
        raise ValueError("O artefato informado não corresponde à classe Stain (ID 5).")
    return artifact


@st.cache_data(show_spinner=False, max_entries=24)
def inspect_image(
    model_path: Path, model_signature: tuple, rgb_path: Path, label_path: Path,
    image_signatures: tuple,
) -> dict:
    """Prediz apenas o RGB selecionado e calcula as métricas com a referência."""
    artifact = load_model(model_path, model_signature)
    rgb = load_image(rgb_path)
    started = perf_counter()
    prediction = predict_regions(artifact, rgb)
    inference_seconds = perf_counter() - started
    reference = create_binary_mask(load_label(label_path), DamageClass.STAIN)
    counts = count_binary_pixels(reference, prediction.mask)
    return {
        "rgb": rgb,
        "reference": reference,
        "prediction": prediction.mask,
        "boundaries": find_boundaries(prediction.regions, mode="inner"),
        "region_count": prediction.region_ids.size,
        "counts": counts,
        "metrics": pixel_metrics(counts),
        "inference_seconds": inference_seconds,
    }


def blend_masks(rgb: np.ndarray, layers: tuple, opacity: float) -> np.ndarray:
    """Aplica cor somente nos pixels das camadas e preserva o restante do RGB."""
    overlay = rgb.copy()
    for mask, color in layers:
        overlay[mask] = np.rint((1 - opacity) * rgb[mask] + opacity * np.asarray(color))
    return overlay


def error_overlay(rgb: np.ndarray, reference: np.ndarray, prediction: np.ndarray,
                  opacity: float) -> np.ndarray:
    """Sobrepõe TP, FP e FN, mantendo os verdadeiros negativos em RGB."""
    real = reference == 1
    predicted = prediction == 1
    layers = (
        (real & predicted, ERROR_COLORS["TP"]),
        (~real & predicted, ERROR_COLORS["FP"]),
        (real & ~predicted, ERROR_COLORS["FN"]),
    )
    return blend_masks(rgb, layers, opacity)
