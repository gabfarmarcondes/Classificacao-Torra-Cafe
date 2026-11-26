"""Etapa 2 - Testes de atalho: a torra pode ser adivinhada sem olhar a cor do grão?

Entrada : outputs/audit/ (unique.csv, folds.json) + outputs/preprocess/ (masks, norm/)
Saídas  : outputs/features/*.npy          vetores por imagem (reaproveitados na etapa 3)
          outputs/shortcuts/per_fold.csv  resultado de cada split
          outputs/shortcuts/summary.csv   média ± desvio por teste (vai para o artigo)

Testes (todos com regressão logística, nos mesmos 15 splits da auditoria):
  so_fundo   6 números do papel em volta do grão     (imagem original e normalizada)
  so_forma   11 números da silhueta, sem cor
  cor_grao   16 números da cor do grão                (referência: a pista "legítima")
  so_forma em pares vizinhos: Green×Light (mesma origem) vs Light×Medium e
  Medium×Dark (origens diferentes) -> separa efeito da torra de efeito da origem.

Leitura: acaso = 25% (4 classes) e 50% (pares). Bem acima do acaso = a pista existe.

Uso (na raiz do projeto): python -m src.shortcuts
     --data padrão = raiz do projeto (onde estão train/ e test/)
     --out  padrão = <raiz>/outputs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import DEFAULT_OUT, ROOT, load_folds, load_rgb, load_unique
from .handcrafted import background_features, color_features, shape_features

PAIRS = [("Green", "Light", "mesma origem"),
         ("Light", "Medium", "origens diferentes"),
         ("Medium", "Dark", "origens diferentes")]


def compute_features(data_dir: Path, out: Path, uniq: pd.DataFrame) -> dict[str, np.ndarray]:
    """Calcula (ou carrega do cache) os vetores de fundo, forma e cor."""
    fdir = out / "features"
    names = ["fundo_original", "fundo_normalizada", "forma", "cor_original", "cor_normalizada"]
    if all((fdir / f"{n}.npy").exists() for n in names):
        return {n: np.load(fdir / f"{n}.npy") for n in names}
    masks = np.load(out / "preprocess" / "masks.npz")["masks"]
    feats = {n: [] for n in names}
    for i, rel in enumerate(uniq.path):
        rgb = load_rgb(data_dir / rel)
        nrm = load_rgb(out / "preprocess" / "norm" / rel)
        m = masks[i]
        feats["fundo_original"].append(background_features(rgb, m))
        feats["fundo_normalizada"].append(background_features(nrm, m))
        feats["forma"].append(shape_features(m))
        feats["cor_original"].append(color_features(rgb, m))
        feats["cor_normalizada"].append(color_features(nrm, m))
    fdir.mkdir(parents=True, exist_ok=True)
    feats = {n: np.asarray(v) for n, v in feats.items()}
    for n, v in feats.items():
        np.save(fdir / f"{n}.npy", v)
    return feats


def run_cv(X, y, folds, keep=None):
    """Regressão logística em cada split. `keep` = máscara de linhas a usar (pares)."""
    rows = []
    for f in folds:
        tr, te = np.asarray(f["train"]), np.asarray(f["test"])
        if keep is not None:
            tr, te = tr[keep[tr]], te[keep[te]]
        if len(np.unique(y[tr])) < 2 or len(te) == 0:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        clf.fit(X[tr], y[tr])
        p = clf.predict(X[te])
        rows.append({"rep": f["rep"], "fold": f["fold"],
                     "acc": accuracy_score(y[te], p),
                     "f1_macro": f1_score(y[te], p, average="macro")})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=ROOT, type=Path,
                    help="pasta com train/ e test/ (padrão: raiz do projeto)")
    ap.add_argument("--out", default=DEFAULT_OUT, type=Path)
    args = ap.parse_args()
    out = args.out / "shortcuts"
    out.mkdir(parents=True, exist_ok=True)

    uniq = load_unique(args.out)
    folds = load_folds(args.out)
    stats = pd.read_csv(args.out / "preprocess" / "stats.csv")
    if stats.path.tolist() != uniq.path.tolist():
        raise SystemExit("unique.csv mudou desde o pré-processamento: rode a etapa 1 de novo "
                         "e apague outputs/features/")
    y = uniq.y.to_numpy()
    F = compute_features(args.data, args.out, uniq)

    tests = [  # (teste, features, descrição)
        ("so_fundo", "fundo_original", "papel em volta do grão, imagem original"),
        ("so_fundo", "fundo_normalizada", "papel em volta do grão, imagem normalizada"),
        ("so_forma", "forma", "silhueta, sem cor"),
        ("cor_grao", "cor_original", "cor do grão (referência)"),
        ("cor_grao", "cor_normalizada", "cor do grão normalizada (referência)"),
    ]
    all_rows = []
    for name, feat, desc in tests:
        for r in run_cv(F[feat], y, folds):
            all_rows.append({"teste": name, "features": feat, "classes": "4 classes",
                             "acaso": 0.25, "descricao": desc, **r})
    labels = uniq.label.to_numpy()
    for a, b, origem in PAIRS:
        keep = np.isin(labels, [a, b])
        for feat in ("forma", "cor_original"):
            for r in run_cv(F[feat], y, folds, keep):
                all_rows.append({"teste": "par_" + ("forma" if feat == "forma" else "cor"),
                                 "features": feat, "classes": f"{a} x {b}", "acaso": 0.5,
                                 "descricao": origem, **r})

    per_fold = pd.DataFrame(all_rows)
    per_fold.to_csv(out / "per_fold.csv", index=False)
    summary = (per_fold.groupby(["teste", "features", "classes", "descricao", "acaso"], sort=False)
               .agg(acc_media=("acc", "mean"), acc_desvio=("acc", "std"),
                    f1_media=("f1_macro", "mean"), f1_desvio=("f1_macro", "std"),
                    n_splits=("acc", "size"))
               .reset_index())
    summary.to_csv(out / "summary.csv", index=False)

    print(f"[atalhos] {len(uniq)} imagens, {len(folds)} splits\n")
    show = summary.copy()
    show["acurácia"] = [f"{m:.1%} ± {s:.1%}" for m, s in zip(show.acc_media, show.acc_desvio)]
    show["acaso"] = [f"{a:.0%}" for a in show.acaso]
    print(show[["teste", "features", "classes", "descricao", "acaso", "acurácia"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
