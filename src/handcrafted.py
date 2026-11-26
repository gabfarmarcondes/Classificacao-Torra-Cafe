"""Descritores "feitos à mão" usados nas etapas 2 (atalhos) e 3 (extração).

Todos recebem a imagem RGB uint8 e a máscara booleana do grão.
"""
from __future__ import annotations

import numpy as np
from skimage import measure, morphology

from .common import chroma, rgb_to_lab


def background_features(rgb: np.ndarray, mask: np.ndarray, ring: int = 12) -> np.ndarray:
    """6 números sobre o PAPEL: média e desvio de L*, a*, b* fora do grão.

    Ignora uma faixa de `ring` px em volta do grão (sombra). Não contém nada
    sobre o grão; se isso classificar a torra, o fundo é uma pista.
    """
    bg = ~morphology.dilation(mask, morphology.disk(ring))
    lab = rgb_to_lab(rgb)[bg]
    return np.r_[lab.mean(0), lab.std(0)]


BACKGROUND_NAMES = ["fundo_L_media", "fundo_a_media", "fundo_b_media",
                    "fundo_L_desvio", "fundo_a_desvio", "fundo_b_desvio"]


def shape_features(mask: np.ndarray) -> np.ndarray:
    """11 números sobre a SILHUETA, sem nenhuma cor.

    área (fração da imagem), razão de aspecto, excentricidade, solidez e os
    7 momentos de Hu em escala log (invariantes a rotação e tamanho).
    """
    props = measure.regionprops(mask.astype(np.uint8))[0]
    aspect = props.axis_major_length / max(props.axis_minor_length, 1e-6)
    hu = props.moments_hu
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    return np.r_[props.area / mask.size, aspect, props.eccentricity, props.solidity, hu_log]


SHAPE_NAMES = ["area", "razao_aspecto", "excentricidade", "solidez"] + [f"hu{i}" for i in range(1, 8)]


def color_features(rgb: np.ndarray, mask: np.ndarray, erode: int = 3) -> np.ndarray:
    """16 números sobre a COR DO GRÃO (só pixels do grão, borda erodida).

    Para L*, a*, b*: média, desvio, percentis 10/50/90 (15) + croma médio (1).
    """
    core = morphology.erosion(mask, morphology.disk(erode)) if erode else mask
    if core.sum() < 50:          # grão minúsculo: usa a máscara inteira
        core = mask
    lab = rgb_to_lab(rgb)
    px = lab[core]
    pct = np.percentile(px, [10, 50, 90], axis=0)          # (3, 3)
    return np.r_[px.mean(0), px.std(0), pct.T.ravel(), chroma(lab)[core].mean()]


COLOR_NAMES = ([f"{c}_media" for c in "Lab"] + [f"{c}_desvio" for c in "Lab"]
               + [f"{c}_p{p}" for c in "Lab" for p in (10, 50, 90)] + ["croma_media"])
