"""Painéis de probabilidade, ocupação e inspeção individual dos superpixels."""

import numpy as np
import streamlit as st
from matplotlib import colormaps
from skimage.segmentation import find_boundaries
from streamlit_image_coordinates import streamlit_image_coordinates

from src.region_features import FEATURE_NAMES
from visualization.inspection import blend_masks, clicked_region, region_counts


REGION_COLOR = (255, 219, 87)
TRAINING_LABELS = {-1: "Ambígua", 0: "Não mancha", 1: "Mancha"}
VIRIDIS = colormaps["viridis"]


def show_heatmaps(data: dict, opacity: float) -> None:
    """Mostra valores por região em viridis com a mesma escala fixa para os dois mapas."""
    st.subheader("Probabilidade e ocupação por região")
    overlay = st.toggle("Sobrepor mapas de calor ao RGB")
    labeling = data["labeling"]
    for column, (title, values) in zip(st.columns(2), (
        ("Probabilidade de Stain", data["probabilities"]),
        ("Ocupação real de Stain", labeling.target_fractions),
    )):
        colored = VIRIDIS(values[labeling.pixel_indices], bytes=True)[..., :3]
        if overlay:
            colored = np.rint((1 - opacity) * data["rgb"] + opacity * colored).astype(np.uint8)
        column.markdown(f"**{title}**")
        column.image(colored, width="stretch")
    scale = np.tile(np.linspace(0, 1, 256), (10, 1))
    st.image(VIRIDIS(scale, bytes=True)[..., :3], width=256, alt="Escala viridis de zero a cem por cento")
    st.caption("0% · 25% · 50% · 75% · 100% — escala fixa para os dois mapas")
    config = data["labeling_config"]
    st.caption(f"Rótulo de treino · até {config['negative_max']:.0%}: não mancha · "
               f"a partir de {config['positive_min']:.0%}: mancha · entre os limites: ambígua.")


def summarize_region(data: dict, predictions: np.ndarray, position: int) -> dict:
    """Resume a região da linha indicada, usando a decisão do limiar ativo."""
    labeled = data["labeling"]
    area = int(labeled.pixel_counts[position])
    predicted = int(predictions[position])
    counts = region_counts(area, int(labeled.target_counts[position]), predicted)
    return {
        "Área (px)": area,
        "Probabilidade de Stain": f"{data['probabilities'][position]:.2%}",
        "Ocupação real": f"{labeled.target_fractions[position]:.2%}",
        "Classe prevista": TRAINING_LABELS[predicted],
        "Rótulo de treino": TRAINING_LABELS[int(labeled.labels[position])],
        **counts,
    }


def highlighted_region(data: dict, region_id: int, *, crop: bool = False) -> np.ndarray:
    """Destaca o contorno e, opcionalmente, recorta o entorno da região."""
    selected = data["regions"] == region_id
    rgb = data["rgb"]
    if crop:
        rows, columns = np.nonzero(selected)
        bounds = np.s_[max(0, rows.min() - 5):rows.max() + 6,
                       max(0, columns.min() - 5):columns.max() + 6]
        selected, rgb = selected[bounds], rgb[bounds]
    # O preenchimento externo falso também marca contornos nas bordas da imagem.
    boundary = find_boundaries(np.pad(selected, 1), mode="inner")[1:-1, 1:-1] & selected
    return blend_masks(rgb, ((boundary, REGION_COLOR),), 1)


def reset_region_selection(image_key: tuple, region_ids: np.ndarray) -> int:
    """Invalida cliques e comparações quando a imagem ou o seu conteúdo muda."""
    if st.session_state.get("region_image_key") != image_key:
        st.session_state.region_image_key = image_key
        st.session_state.region_generation = st.session_state.get("region_generation", 0) + 1
        st.session_state.superpixel_id = int(region_ids[0])
        st.session_state.compared_regions = []
        st.session_state.last_region_click = None
    return st.session_state.region_generation


def show_region_inspector(data: dict, predictions: np.ndarray, generation: int) -> None:
    """Sincroniza a seleção por clique e por ID e apresenta os valores da região."""
    image_panel, details = st.columns([3, 2])
    selected = st.session_state.superpixel_id
    with image_panel:
        st.caption("Clique no RGB para selecionar um superpixel.")
        click = streamlit_image_coordinates(
            highlighted_region(data, selected), width="content", cursor="crosshair",
            key=f"superpixel_click_{generation}", png_compression_level=1,
        )
        if click and click != st.session_state.last_region_click:
            st.session_state.last_region_click = click
            clicked_id = clicked_region(data["regions"], click)
            if clicked_id is not None and clicked_id != selected:
                st.session_state.superpixel_id = clicked_id
                st.rerun()

    with details:
        selected = st.selectbox("ID do superpixel", data["region_ids"].tolist(), key="superpixel_id")
        position = int(np.searchsorted(data["region_ids"], selected))
        summary = summarize_region(data, predictions, position)
        st.table({"Indicador": list(summary), "Valor": [str(value) for value in summary.values()]})
        compared = st.session_state.compared_regions
        if st.button("Adicionar à comparação", disabled=selected in compared):
            st.session_state.compared_regions = [*compared, selected]
        st.dataframe({"Característica": FEATURE_NAMES, "Valor": data["features"][position]},
                     hide_index=True, width="stretch")


def clear_comparison() -> None:
    st.session_state.compared_regions = []


def show_region_comparison(data: dict, predictions: np.ndarray) -> None:
    """Compara contornos, indicadores e características na escala original."""
    st.button("Limpar comparação", on_click=clear_comparison,
              disabled=not st.session_state.compared_regions)
    selected = st.multiselect("Superpixels comparados", data["region_ids"].tolist(),
                             key="compared_regions", help="Remova uma região pelo × ao lado do ID.")
    if len(selected) < 2:
        st.info("Selecione pelo menos dois superpixels para comparar.")
    if not selected:
        return

    positions = np.searchsorted(data["region_ids"], selected)
    summaries = [summarize_region(data, predictions, int(position)) for position in positions]
    for start in range(0, len(selected), 3):
        group = selected[start:start + 3]
        for column, region_id in zip(st.columns(len(group)), group):
            column.image(highlighted_region(data, region_id, crop=True),
                         caption=f"Superpixel {region_id}", width="content")
    indicators = {"Indicador": list(summaries[0])}
    features = {"Característica": FEATURE_NAMES}
    for region_id, position, summary in zip(selected, positions, summaries):
        title = f"ID {region_id}"
        indicators[title] = [str(value) for value in summary.values()]
        features[title] = data["features"][position]
    st.dataframe(indicators, hide_index=True, width="stretch")
    st.dataframe(features, hide_index=True, width="stretch")
