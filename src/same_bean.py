"""Etapa 3b - Detecção automática do mesmo grão fotografado mais de uma vez.

A auditoria (etapa 0) só detecta a MESMA FOTO. O mesmo grão fotografado de novo
(girado, deslocado, com outra luz) gera pixels diferentes e passa despercebido. Se
ele cair no treino e no teste, o modelo pode acertar por reconhecer aquele grão.

Como funciona (sem conferência manual):
  1. Candidatos: cada imagem sem_fundo passa pelo DINOv2 em 8 orientações; para cada
     imagem, os --k vizinhos mais parecidos da MESMA classe viram pares candidatos.
  2. Verificação geométrica: em cada par, pontos-chave SIFT detectados só dentro do
     grão são ligados entre as duas imagens, e o RANSAC conta quantas ligações são
     explicadas por UMA mesma transformação (giro + deslocamento + zoom entre 0,6x e
     1,6x; fora disso a transformação é degenerada e descartada). Mesmo grão
     = dezenas de ligações coerentes (rachaduras, manchas, borda); grãos diferentes =
     poucas, ao acaso.
  3. Limiar calibrado nos dados: pares de CLASSES DIFERENTES são, com certeza, grãos
     diferentes. Para cada imagem, os --k-neg vizinhos mais parecidos de outra classe
     são verificados do mesmo jeito; o limiar é o maior número de ligações entre eles
     + --margem (1).
  4. Critério liberal, de propósito: no dataset completo a distribuição dos pares da
     mesma classe é contínua (sem um vão entre "mesmo grão" e "grão diferente"), e a
     inspeção visual não resolve o meio da faixa. Como agrupar dois grãos diferentes
     por engano não prejudica nada (só os mantém do mesmo lado nos folds) e deixar
     passar um grão repetido é vazamento, tudo acima do que grãos distintos atingem
     por acaso é tratado como mesmo grão. O resumo registra também quantos pares
     haveria com critérios mais rígidos, para reportar a faixa.
  5. Pares candidatos com ligações >= limiar são marcados como mesmo grão.

Saídas: meta/mesmo_grao.csv                    pares candidatos, ligações e decisão
                                               (versionado no git; a auditoria lê)
        outputs/same_bean/confirmados.png      pares marcados, com as ligações desenhadas
        outputs/same_bean/perto_do_limiar.png  rejeitados mais próximos do limiar
        outputs/same_bean/ligacoes.png         distribuição: candidatos x classes diferentes
        outputs/same_bean/resumo.json          limiar e contagens (vão para o artigo)
        outputs/same_bean/dinov2_d8.npy        embeddings (cache)

Uso (na raiz do projeto): python -m src.same_bean
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from skimage import morphology

from .common import DEFAULT_OUT, ROOT, load_rgb, load_unique, save_json
from .deep_features import Backbone, to_tensor
from .image_versions import load_version

CONFIRM_FILE = ROOT / "meta" / "mesmo_grao.csv"


# ------------------------------------------------------------ candidatos (DINOv2)
def d8(im: np.ndarray, k: int) -> np.ndarray:
    """Uma das 8 orientações: k=0..3 rotações de 90°, k=4..7 o mesmo após espelhar."""
    if k >= 4:
        im = im[:, ::-1]
    return np.ascontiguousarray(np.rot90(im, k % 4))


def embed_d8(model, images, device: str, batch: int) -> np.ndarray:
    """(N, 8, D) embeddings com norma 1."""
    out = []
    for k in range(8):
        feats, buf = [], []
        for im in images:
            buf.append(d8(im, k))
            if len(buf) == batch:
                feats.append(model(to_tensor(buf).to(device)).cpu().numpy())
                buf = []
        if buf:
            feats.append(model(to_tensor(buf).to(device)).cpu().numpy())
        out.append(np.concatenate(feats))
    E = np.stack(out, axis=1).astype(np.float32)
    return E / np.linalg.norm(E, axis=-1, keepdims=True)


def similarity(E: np.ndarray) -> np.ndarray:
    """S[i, j] = maior cosseno entre i e qualquer orientação de j (simetrizada)."""
    n, t, d = E.shape
    S = (E[:, 0, :] @ E.reshape(n * t, d).T).reshape(n, n, t).max(-1)
    S = np.maximum(S, S.T)
    np.fill_diagonal(S, -np.inf)
    return S


def neighbor_pairs(S, labels, k, same: bool) -> set[tuple[int, int]]:
    """Para cada imagem, os k vizinhos mais parecidos da mesma classe (ou de outra)."""
    cls = labels[:, None] == labels[None, :]
    allowed = cls if same else ~cls
    Sm = np.where(allowed, S, -np.inf)
    pairs = set()
    for i in range(len(S)):
        for j in np.argsort(-Sm[i])[:k]:
            if np.isfinite(Sm[i, j]):
                pairs.add((min(i, int(j)), max(i, int(j))))
    return pairs


# ------------------------------------------------------------ verificação (SIFT)
class Verifier:
    """Conta ligações SIFT geometricamente coerentes entre dois grãos."""

    def __init__(self, data_dir: Path, paths, masks, reproj: float = 4.0, ratio: float = 0.8,
                 min_scale: float = 0.6, max_scale: float = 1.6):
        self.data_dir, self.paths, self.masks = data_dir, list(paths), masks
        self.min_scale, self.max_scale = min_scale, max_scale
        self.sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.02)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.reproj, self.ratio = reproj, ratio
        self.cache = {}

    def features(self, i: int, flip: bool):
        key = (i, flip)
        if key not in self.cache:
            gray = cv2.cvtColor(load_rgb(self.data_dir / self.paths[i]), cv2.COLOR_RGB2GRAY)
            mask = morphology.dilation(self.masks[i], morphology.disk(2)).astype(np.uint8) * 255
            if flip:
                gray, mask = gray[:, ::-1].copy(), mask[:, ::-1].copy()
            self.cache[key] = self.sift.detectAndCompute(gray, mask)
        return self.cache[key]

    def match(self, i: int, j: int):
        """Maior nº de ligações coerentes (B normal ou espelhada) e os detalhes."""
        best = (0, None)
        ka, da = self.features(i, False)
        if da is None or len(ka) < 3:
            return best
        for flip in (False, True):
            kb, db = self.features(j, flip)
            if db is None or len(kb) < 3:
                continue
            good = [m for m, n in (p for p in self.matcher.knnMatch(da, db, k=2) if len(p) == 2)
                    if m.distance < self.ratio * n.distance]
            if len(good) < 4:
                continue
            src = np.float32([ka[g.queryIdx].pt for g in good])
            dst = np.float32([kb[g.trainIdx].pt for g in good])
            M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                 ransacReprojThreshold=self.reproj)
            if M is None or inl is None:       # RANSAC não achou transformação
                continue
            scale = float(np.hypot(M[0, 0], M[1, 0]))
            if not (self.min_scale <= scale <= self.max_scale):
                continue                      # transformação degenerada (ex.: encolhe tudo num ponto)
            kept = [g for g, keep in zip(good, inl.ravel()) if keep]
            # conta pontos distintos nos dois lados (evita vários pontos ligados ao mesmo)
            n = min(len({g.queryIdx for g in kept}), len({g.trainIdx for g in kept}))
            if n > best[0]:
                best = (n, {"flip": flip, "M": M, "matches": kept, "kb": kb, "ka": ka})
        return best


# ------------------------------------------------------------ figuras
def draw_pairs(ver: Verifier, rows, path, size=224):
    if not rows:
        return
    panels = []
    for r in rows:
        n, det = ver.match(r["i"], r["j"])
        A = load_rgb(ver.data_dir / ver.paths[r["i"]])
        B = load_rgb(ver.data_dir / ver.paths[r["j"]])
        if det and det["flip"]:
            B = B[:, ::-1].copy()
        if det:
            vis = cv2.drawMatches(A, det["ka"], B, det["kb"], det["matches"], None,
                                  matchColor=(255, 0, 0), flags=2)
        else:
            vis = np.hstack([A, B])
        im = Image.fromarray(vis)
        canvas = Image.new("RGB", (im.width + 190, im.height), "white")
        canvas.paste(im, (0, 0))
        d = ImageDraw.Draw(canvas)
        d.text((im.width + 8, 10), f"#{r['n']}  {r['classe']}", fill="black")
        d.text((im.width + 8, 30), f"ligações: {n}", fill="black")
        d.text((im.width + 8, 50), f"DINOv2: {r['similaridade']:.3f}", fill="black")
        d.text((im.width + 8, 80), Path(r["path_a"]).name, fill="gray")
        d.text((im.width + 8, 98), Path(r["path_b"]).name, fill="gray")
        panels.append(canvas)
    sheet = Image.new("RGB", (panels[0].width, sum(p.height for p in panels)), "white")
    y = 0
    for p in panels:
        sheet.paste(p, (0, y))
        y += p.height
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def plot_inliers(cand, neg, thr, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    top = max(max(cand, default=0), max(neg, default=0)) + 2
    bins = np.arange(0, top + 1)
    ax.hist(neg, bins=bins, alpha=0.6, label="classes diferentes (com certeza grãos distintos)")
    ax.hist(cand, bins=bins, alpha=0.6, label="candidatos da mesma classe")
    ax.axvline(thr - 0.5, color="k", ls="--", lw=1, label=f"limiar ({thr} ligações)")
    ax.set_yscale("log")
    ax.set_xlabel("ligações SIFT geometricamente coerentes")
    ax.set_ylabel("nº de pares (escala log)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=ROOT, type=Path,
                    help="pasta com train/ e test/ (padrão: raiz do projeto)")
    ap.add_argument("--out", default=DEFAULT_OUT, type=Path)
    ap.add_argument("--k", type=int, default=10, help="vizinhos da mesma classe verificados por imagem")
    ap.add_argument("--k-neg", type=int, default=5, help="vizinhos de outra classe usados na calibração")
    ap.add_argument("--margem", type=int, default=1,
                    help="somada ao máximo dos pares de classes diferentes para formar o limiar "
                         "(1 = critério liberal: tudo acima do que grãos distintos atingem por acaso)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--random-weights", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    out = args.out / "same_bean"
    out.mkdir(parents=True, exist_ok=True)

    uniq = load_unique(args.out)
    masks = np.load(args.out / "preprocess" / "masks.npz")["masks"]
    labels = uniq.label.to_numpy()

    # 1) candidatos pelo DINOv2
    cache = out / "dinov2_d8.npy"
    if cache.exists() and not args.random_weights and np.load(cache, mmap_mode="r").shape[0] == len(uniq):
        E = np.load(cache)
    else:
        images = [load_version("sem_fundo", p, masks[i], args.data, args.out)
                  for i, p in enumerate(uniq.path)]
        model = Backbone("dinov2_small", pretrained=not args.random_weights).to(args.device)
        E = embed_d8(model, images, args.device, args.batch)
        if not args.random_weights:
            np.save(cache, E)
    S = similarity(E)
    # pares já agrupados continuam na lista: assim a decisão é estável entre execuções
    cand = sorted(neighbor_pairs(S, labels, args.k, same=True))
    neg = sorted(neighbor_pairs(S, labels, args.k_neg, same=False))
    print(f"[mesmo grão] {len(cand)} pares candidatos (mesma classe) e "
          f"{len(neg)} pares de calibração (classes diferentes)")

    # 2) verificação geométrica
    ver = Verifier(args.data, uniq.path, masks)
    neg_inl = [ver.match(i, j)[0] for i, j in neg]
    thr = int(max(neg_inl, default=0)) + args.margem
    rows = []
    for i, j in cand:
        n, _ = ver.match(i, j)
        rows.append({"i": i, "j": j, "path_a": uniq.path[i], "path_b": uniq.path[j],
                     "classe": labels[i], "similaridade": round(float(S[i, j]), 4),
                     "ligacoes": n, "mesmo_grao": "s" if n >= thr else ""})
    table = pd.DataFrame(rows).sort_values(["ligacoes", "similaridade"], ascending=False).reset_index(drop=True)
    table.insert(0, "n", np.arange(1, len(table) + 1))

    # 3) grava as decisões
    CONFIRM_FILE.parent.mkdir(parents=True, exist_ok=True)
    table.drop(columns=["i", "j"]).to_csv(CONFIRM_FILE, index=False)

    # 4) figuras e resumo
    confirmed = table[table.mesmo_grao == "s"]
    near = table[table.mesmo_grao != "s"].head(8)
    draw_pairs(ver, confirmed.head(40).to_dict("records"), out / "confirmados.png")
    draw_pairs(ver, near.to_dict("records"), out / "perto_do_limiar.png")
    plot_inliers(table.ligacoes.tolist(), neg_inl, thr, out / "ligacoes.png")
    for old_fig in ("pares.png", "similaridades.png"):
        (out / old_fig).unlink(missing_ok=True)

    resumo = {
        "candidatos_verificados": len(cand),
        "pares_calibracao_classes_diferentes": len(neg),
        "max_ligacoes_classes_diferentes": int(max(neg_inl, default=0)),
        "limiar_ligacoes": thr,
        "pares_mesmo_grao": int(len(confirmed)),
        "pares_por_limiar": {f">={v}": int((table.ligacoes >= v).sum())
                             for v in sorted({thr, thr + 4, thr + 9, 25})},
        "pares_mesmo_grao_por_classe": confirmed.classe.value_counts().to_dict(),
        "imagens_envolvidas": int(len(set(confirmed.path_a) | set(confirmed.path_b))),
        "maior_ligacao_rejeitada": int(near.ligacoes.max()) if len(near) else 0,
        "pares_zona_intermediaria": int(((table.ligacoes > max(neg_inl, default=0))
                                         & (table.ligacoes < thr)).sum()),
        "pesos_dinov2": "ALEATÓRIOS (teste de código)" if args.random_weights else "facebook/dinov2-small",
    }
    save_json(resumo, out / "resumo.json")
    print(f"[mesmo grão] classes diferentes: no máximo {resumo['max_ligacoes_classes_diferentes']} "
          f"ligações -> limiar = {thr}")
    print(f"[mesmo grão] pares do mesmo grão: {resumo['pares_mesmo_grao']} "
          f"{resumo['pares_mesmo_grao_por_classe']} ({resumo['imagens_envolvidas']} imagens)")
    print(f"[mesmo grão] sensibilidade (pares por limiar de ligações): {resumo['pares_por_limiar']}")
    print(f"[mesmo grão] decisões em {CONFIRM_FILE.relative_to(ROOT)}; "
          f"rode python -m src.audit para atualizar os folds")


if __name__ == "__main__":
    main()
