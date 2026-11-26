"""Etapa 1 - Máscara do grão e normalização de cor pelo fundo.

Entrada : outputs/audit/unique.csv + imagens em train/ e test/ (raiz do projeto)
Saídas  : outputs/preprocess/
            masks.npz            máscaras booleanas (N, 224, 224), na ordem de unique.csv
            norm/<path>.png      imagens normalizadas (mesma estrutura de pastas)
            stats.csv            por imagem: fração de grão, cor do fundo (p90), ganhos
            qc_<classe>.png      contornos das máscaras + antes/depois, para conferir a olho

Como funciona (resumo):
  Máscara (--method):
    u2net (padrão)  segmentador pré-treinado de "objeto em destaque" (U²-Net,
                    via rembg). Separa o grão da sombra pela forma, não só pela
                    cor. Testado: limiar de cor falha nos grãos escuros (a sombra
                    de contato é tão escura quanto o grão e os reflexos são
                    claros e sem cor como o fundo).
    threshold       alternativa sem rede: grão = croma > --chroma-thr OU
                    L* < --dark-thr, seguido de abertura, fechamento e maior
                    componente. Funciona bem no Green/Light, pega sombra no Dark.
    Nos dois casos: fica só o maior componente conectado e os buracos são preenchidos.
  Normalização: o fundo (excluindo uma faixa de --bg-ring px em volta do grão,
    onde fica a sombra) deveria ser branco. O percentil --bg-pct de cada canal
    RGB nesse fundo é levado a 255 multiplicando o canal por um ganho.

Uso (na raiz do projeto): python -m src.preprocess
     --data padrão = raiz do projeto (onde estão train/ e test/)
     --out  padrão = <raiz>/outputs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_fill_holes
from skimage import measure, morphology, segmentation

from .common import (CLASSES, DEFAULT_OUT, ROOT, chroma, load_rgb, load_unique,
                     rgb_to_lab)


def largest_component(m: np.ndarray) -> np.ndarray:
    """Mantém só a maior mancha conectada e preenche buracos internos."""
    lbl = measure.label(m)
    if lbl.max() == 0:                       # nada encontrado: máscara vazia
        return np.zeros(m.shape, bool)
    largest = np.argmax(np.bincount(lbl.ravel())[1:]) + 1
    return binary_fill_holes(lbl == largest)


def mask_threshold(rgb: np.ndarray, chroma_thr: float = 10.0, dark_thr: float = 45.0,
                   open_r: int = 3, close_r: int = 5) -> np.ndarray:
    """Máscara por limiar em Lab (alternativa sem rede neural)."""
    lab = rgb_to_lab(rgb)
    raw = (chroma(lab) > chroma_thr) | (lab[..., 0] < dark_thr)
    m = morphology.opening(raw, morphology.disk(open_r))
    m = morphology.closing(m, morphology.disk(close_r))
    return largest_component(m)


class U2NetMasker:
    """Máscara com U²-Net (rembg). O modelo (~176 MB) é baixado na 1ª execução."""

    def __init__(self):
        from rembg import new_session  # import tardio: só exige rembg se for usado
        self.session = new_session("u2net")

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        from rembg import remove
        alpha = np.asarray(remove(Image.fromarray(rgb), session=self.session, only_mask=True))
        return largest_component(alpha > 127)


def normalize_by_background(rgb: np.ndarray, mask: np.ndarray, pct: float = 90.0,
                            ring: int = 0):
    """White patch usando os pixels de fundo como referência de branco.

    ring > 0 ignora uma faixa de `ring` px em volta do grão (onde fica a sombra).
    Retorna (imagem normalizada uint8, referência RGB, ganhos RGB).
    """
    bg = ~mask
    if ring > 0:
        bg &= ~morphology.dilation(mask, morphology.disk(ring))
    ref = np.percentile(rgb[bg].reshape(-1, 3).astype(np.float64), pct, axis=0)
    gains = 255.0 / np.maximum(ref, 1.0)
    norm = np.clip(rgb.astype(np.float64) * gains, 0, 255).round().astype(np.uint8)
    return norm, ref, gains


def qc_sheet(data_dir, norm_dir, uniq, masks, cls, path, n=8, size=150, seed=0):
    """Para n imagens da classe: original com contorno | normalizada."""
    idx = uniq.index[uniq.label == cls].to_numpy()
    if len(idx) == 0:
        return
    rng = np.random.default_rng(seed)
    idx = rng.choice(idx, size=min(n, len(idx)), replace=False)
    sheet = Image.new("RGB", (2 * size, len(idx) * size), "white")
    for r, i in enumerate(idx):
        rgb = load_rgb(data_dir / uniq.path[i])
        over = segmentation.mark_boundaries(rgb, masks[i], color=(1, 0, 0), mode="thick")
        over = (over * 255).astype(np.uint8)
        nrm = load_rgb(norm_dir / uniq.path[i])
        for c, im in enumerate((over, nrm)):
            sheet.paste(Image.fromarray(im).resize((size - 4, size - 4)), (c * size + 2, r * size + 2))
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=ROOT, type=Path,
                    help="pasta com train/ e test/ (padrão: raiz do projeto)")
    ap.add_argument("--out", default=DEFAULT_OUT, type=Path)
    ap.add_argument("--method", choices=["u2net", "threshold"], default="u2net")
    ap.add_argument("--chroma-thr", type=float, default=10.0, help="só para --method threshold")
    ap.add_argument("--dark-thr", type=float, default=45.0, help="só para --method threshold")
    ap.add_argument("--bg-pct", type=float, default=90.0)
    ap.add_argument("--bg-ring", type=int, default=12,
                    help="px em volta do grão ignorados no fundo (região da sombra)")
    args = ap.parse_args()
    out = args.out / "preprocess"
    norm_dir = out / "norm"
    out.mkdir(parents=True, exist_ok=True)

    if args.method == "u2net":
        get_mask = U2NetMasker()
    else:
        get_mask = lambda rgb: mask_threshold(rgb, args.chroma_thr, args.dark_thr)  # noqa: E731

    uniq = load_unique(args.out)
    masks, rows = [], []
    for i, rel in enumerate(uniq.path):
        rgb = load_rgb(args.data / rel)
        m = get_mask(rgb)
        norm, ref, gains = normalize_by_background(rgb, m, args.bg_pct, args.bg_ring)
        dst = norm_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(norm).save(dst)
        masks.append(m)
        touches = bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())
        rows.append({"path": rel, "label": uniq.label[i], "frac_grao": float(m.mean()),
                     "toca_borda": touches,
                     "fundo_R": ref[0], "fundo_G": ref[1], "fundo_B": ref[2],
                     "ganho_R": gains[0], "ganho_G": gains[1], "ganho_B": gains[2]})
        if (i + 1) % 100 == 0:
            print(f"[pré-processamento] {i + 1}/{len(uniq)}")

    masks = np.stack(masks)
    np.savez_compressed(out / "masks.npz", masks=masks)
    stats = pd.DataFrame(rows)
    stats.to_csv(out / "stats.csv", index=False)
    for cls in CLASSES:
        qc_sheet(args.data, norm_dir, uniq, masks, cls, out / f"qc_{cls}.png")

    # alertas: máscaras suspeitas para olhar primeiro
    sus = stats[(stats.frac_grao < 0.05) | (stats.frac_grao > 0.6) | stats.toca_borda]
    print(f"[pré-processamento] {len(uniq)} imagens processadas")
    print("[pré-processamento] fração média de grão por classe:")
    print(stats.groupby("label").frac_grao.mean().round(3).to_string())
    print("[pré-processamento] cor do fundo (p%d) média por classe:" % args.bg_pct)
    print(stats.groupby("label")[["fundo_R", "fundo_G", "fundo_B"]].mean().round(1).to_string())
    print(f"[pré-processamento] máscaras suspeitas (vazias, enormes ou tocando a borda): {len(sus)}")
    if len(sus):
        sus.to_csv(out / "suspeitas.csv", index=False)
        print(f"   -> lista em {out / 'suspeitas.csv'}")


if __name__ == "__main__":
    main()
