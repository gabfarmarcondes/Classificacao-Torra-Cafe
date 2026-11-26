"""Versões de cada imagem usadas na extração (etapa 3) e na interpretação (etapa 8).

  original     a foto como está no dataset
  normalizada  iluminação corrigida pelo fundo (gerada na etapa 1)
  sem_fundo    grão original sobre branco puro: elimina a pista do fundo por construção
  so_fundo     disco de raio fixo em volta do grão apagado e preenchido com o papel:
               só sobra o fundo, sem tamanho nem formato do grão. Serve de controle:
               se um backbone acerta a torra nessa versão, ele consegue usar o fundo.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import load_rgb

VERSIONS = ["original", "normalizada", "sem_fundo", "so_fundo"]


def remove_background(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Grão original sobre fundo branco (255)."""
    out = np.full_like(rgb, 255)
    out[mask] = rgb[mask]
    return out


def bean_radius(mask: np.ndarray) -> float:
    """Maior distância entre o centro do grão e um pixel do grão."""
    ys, xs = np.nonzero(mask)
    return float(np.sqrt((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2).max())


def erase_radius(masks: np.ndarray, margin: int = 8) -> int:
    """Raio do disco de apagamento: cobre o maior grão do dataset + margem."""
    return int(np.ceil(max(bean_radius(m) for m in masks))) + margin


def erase_bean(rgb: np.ndarray, mask: np.ndarray, radius: int, ring: int = 6) -> np.ndarray:
    """Apaga um disco de raio FIXO centrado no grão e preenche com o papel.

    O disco tem o mesmo tamanho em todas as imagens (cobre o maior grão do
    dataset), então nem o tamanho nem o formato do grão ficam visíveis; a maior
    parte da sombra também some. O preenchimento é um polinômio de grau 2 em
    (x, y), por canal, ajustado aos pixels de papel fora do disco: continua o
    gradiente de luz sem criar manchas.
    """
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    region = dist <= radius
    paper = dist > radius + ring
    if paper.sum() < 200:                      # disco cobre quase tudo: usa a borda da imagem
        paper = ~region
    xn, yn = xx / w - 0.5, yy / h - 0.5
    design = np.stack([np.ones_like(xn), xn, yn, xn * yn, xn ** 2, yn ** 2], axis=-1)
    out = rgb.astype(np.float64).copy()
    for c in range(3):
        coef, *_ = np.linalg.lstsq(design[paper], out[..., c][paper], rcond=None)
        out[..., c][region] = design[region] @ coef
    return np.clip(out, 0, 255).round().astype(np.uint8)


def load_version(version: str, rel_path: str, mask: np.ndarray,
                 data_dir: Path, out_dir: Path, radius: int | None = None) -> np.ndarray:
    """Carrega a imagem `rel_path` na versão pedida (uint8, H x W x 3)."""
    if version == "original":
        return load_rgb(data_dir / rel_path)
    if version == "normalizada":
        return load_rgb(out_dir / "preprocess" / "norm" / rel_path)
    if version == "sem_fundo":
        return remove_background(load_rgb(data_dir / rel_path), mask)
    if version == "so_fundo":
        if radius is None:
            raise ValueError("so_fundo precisa do raio do disco (erase_radius das máscaras)")
        return erase_bean(load_rgb(data_dir / rel_path), mask, radius)
    raise ValueError(f"versão desconhecida: {version} (use uma de {VERSIONS})")
