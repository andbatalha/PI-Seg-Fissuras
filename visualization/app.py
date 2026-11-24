"""Interface local de inspeção das máscaras de Stain no BFDD."""

from pathlib import Path
from random import choice

import numpy as np
import streamlit as st

from visualization.inspection import (
    blend_masks, error_overlay, file_signature, inspect_image, load_catalog, load_model,
)


ROOT = Path(__file__).resolve().parents[1]
# O caminho relativo evita falhas do OpenCV em caminhos Unicode no Windows.
DATASET = Path("data") / "BFDD"
RESULTS = ROOT / "results" / "region_full"
SORT_FIELDS = {
    "IoU": ("metrics", "IoU"),
    "F1-score": ("metrics", "F1"),
    "Precision": ("metrics", "Precision"),
    "Recall": ("metrics", "Recall"),
    "Falsos positivos": ("counts", "FP"),
    "Falsos negativos": ("counts", "FN"),
}


def select_image(image_id: str) -> None:
    """Atualiza a seleção do menu por meio dos botões de navegação."""
    st.session_state.image_id = image_id


def random_image(image_ids: list[str], current: str) -> None:
    """Sorteia outra imagem entre as opções visíveis."""
    select_image(choice([name for name in image_ids if name != current]))


def navigation(subsets: dict, presence: dict, report: dict) -> str | None:
    """Aplica os filtros e oferece navegação apenas entre IDs permitidos."""
    subset = st.radio("Conjunto", ("Treinamento", "Validação"), index=1, horizontal=True)
    image_ids = sorted(subsets["train" if subset == "Treinamento" else "validation"])
    presence_filter = st.selectbox("Presença de Stain", ("Todas", "Com Stain", "Sem Stain"))
    if presence_filter != "Todas":
        has_stain = presence_filter == "Com Stain"
        unknown = sum(name not in presence for name in image_ids)
        image_ids = [name for name in image_ids if presence.get(name) is has_stain]
        if unknown:
            st.caption(f"{unknown} imagens sem informação de presença ficaram fora do filtro.")

    if subset == "Validação":
        options = ("ID", *SORT_FIELDS) if report else ("ID",)
        sort_by = st.selectbox("Ordenar por", options)
        if sort_by in SORT_FIELDS:
            direction = st.radio("Ordem", ("Maior primeiro", "Menor primeiro"), horizontal=True)
            group, key = SORT_FIELDS[sort_by]
            sign = -1 if direction == "Maior primeiro" else 1
            image_ids.sort(key=lambda name: (
                name not in report, sign * report[name][group][key] if name in report else 0, name,
            ))
        if not report:
            st.caption("O relatório por imagem não está disponível para ordenação por métricas.")

    if not image_ids:
        st.info("Nenhuma imagem corresponde aos filtros selecionados.")
        return None
    if st.session_state.get("image_id") not in image_ids:
        select_image(image_ids[0])
    image_id = st.selectbox("ID da imagem", image_ids, key="image_id")
    position = image_ids.index(image_id)
    st.caption(f"Imagem {position + 1} de {len(image_ids)}")
    previous, following = st.columns(2)
    previous.button("Anterior", on_click=select_image, args=(image_ids[max(0, position - 1)],),
                    disabled=position == 0, width="stretch")
    following.button("Próxima", on_click=select_image,
                     args=(image_ids[min(len(image_ids) - 1, position + 1)],),
                     disabled=position == len(image_ids) - 1, width="stretch")
    st.button("Imagem aleatória", on_click=random_image, args=(image_ids, image_id),
              disabled=len(image_ids) == 1, width="stretch")
    return image_id


def show_comparison(data: dict, show_overlays: bool, show_boundaries: bool, opacity: float) -> None:
    """Mostra o RGB, as duas máscaras e o mapa de erros em duas linhas."""
    rgb, reference, prediction = (data[key] for key in ("rgb", "reference", "prediction"))
    errors = error_overlay(rgb, reference, prediction, opacity)
    if show_overlays:
        target_panel = blend_masks(rgb, ((reference == 1, (41, 105, 255)),), opacity)
        prediction_panel = blend_masks(rgb, ((prediction == 1, (0, 200, 83)),), opacity)
    else:
        target_panel, prediction_panel = reference * 255, prediction * 255

    pairs = (
        (("Imagem RGB", rgb), ("Ground Truth · Stain", target_panel)),
        (("Predição · Random Forest", prediction_panel), ("Mapa de erros", errors)),
    )
    for row in pairs:
        for column, (title, image) in zip(st.columns(2), row):
            if show_boundaries:
                boundary_rgb = np.repeat(image[..., None], 3, axis=-1) if image.ndim == 2 else image
                image = blend_masks(boundary_rgb, ((data["boundaries"], (255, 219, 87)),), 1)
            with column:
                st.markdown(f"**{title}**")
                st.image(image, width="stretch")
    st.caption("🟢 TP · 🔴 FP · 🔵 FN · TN preserva o RGB · 🟡 fronteiras SLIC quando ativadas")


def show_metrics(data: dict, recorded: dict | None) -> None:
    """Mostra as métricas calculadas para a máscara que está na tela."""
    st.subheader("Métricas por pixel")
    for column, (name, title) in zip(st.columns(5), (
        ("Accuracy", "Accuracy"), ("Precision", "Precision"), ("Recall", "Recall"),
        ("IoU", "IoU"), ("F1", "F1-score"),
    )):
        column.metric(title, f"{data['metrics'][name]:.2%}")
    counts = data["counts"]
    for column, name in zip(st.columns(4), ("TP", "TN", "FP", "FN")):
        column.metric(name, f"{counts[name]:,}".replace(",", "."))
    pixels = sum(counts.values())
    real_fraction = (counts["TP"] + counts["FN"]) / pixels
    predicted_fraction = (counts["TP"] + counts["FP"]) / pixels
    st.caption(
        f"Stain na referência {real_fraction:.2%} · Stain previsto {predicted_fraction:.2%} · "
        f"{data['region_count']} regiões SLIC · inferência {data['inference_seconds']:.3f} s "
        "(tempo da execução que gerou a máscara, reutilizado em cache)"
    )
    if recorded is not None and counts != recorded["counts"]:
        st.warning("As contagens atuais diferem do relatório salvo para este ID.")


def show_model(artifact: dict, model_path: Path) -> None:
    """Exibe a configuração que acompanha o estimador salvo."""
    with st.expander("Informações do modelo"):
        st.code(model_path.relative_to(ROOT).as_posix(), language=None)
        slic, forest = st.columns(2)
        with slic:
            st.markdown("**SLIC**")
            st.json(artifact["pipeline_config"]["slic"])
        with forest:
            st.markdown("**Random Forest**")
            st.json(artifact["forest_config"])
        names = artifact["feature_names"]
        st.markdown(f"**{len(names)} características** · {', '.join(names)}")
        st.caption("Classificação pelo predict() do Random Forest, sem limiar ajustável na interface.")


def main() -> None:
    st.set_page_config(page_title="BFDD · Inspeção de Stain", layout="wide")
    st.title("Inspeção de Stain")
    st.caption("BFDD · classe 5 · segmentação por regiões SLIC")
    model_path = RESULTS / "random_forest.joblib"
    model_signature = file_signature(model_path)
    if model_signature is None or not (RESULTS / "split.json").is_file():
        st.error("Coloque random_forest.joblib e split.json em results/region_full/.")
        return
    signatures = tuple(file_signature(RESULTS / name) for name in (
        "split.json", "pixel_validation_per_image.json", "train.npz", "validation.npz",
    ))
    try:
        subsets, presence, report = load_catalog(RESULTS, signatures)
        artifact = load_model(model_path, model_signature)
    except (OSError, ValueError, KeyError, ImportError) as error:
        st.error(f"Não foi possível carregar os artefatos: {error}")
        return

    with st.sidebar:
        st.header("Navegação")
        image_id = navigation(subsets, presence, report)
        st.header("Visualização")
        show_overlays = st.toggle("Sobrepor máscaras ao RGB")
        show_boundaries = st.toggle("Mostrar fronteiras SLIC")
        opacity = st.slider("Opacidade das sobreposições", 0, 100, 65) / 100
    if image_id is not None:
        rgb_path = DATASET / "RGB" / f"{image_id}.JPG"
        label_path = DATASET / "Label" / f"{image_id}.png"
        try:
            with st.spinner("Carregando a imagem e a predição…"):
                data = inspect_image(
                    model_path, model_signature, rgb_path, label_path,
                    (file_signature(rgb_path), file_signature(label_path)),
                )
        except (OSError, ValueError, KeyError) as error:
            st.error(f"Não foi possível inspecionar {image_id}: {error}")
        else:
            st.subheader(image_id)
            show_comparison(data, show_overlays, show_boundaries, opacity)
            show_metrics(data, report.get(image_id))
    show_model(artifact, model_path)


if __name__ == "__main__":
    main()
