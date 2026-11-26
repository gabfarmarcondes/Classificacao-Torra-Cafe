"""Etapa 0 - Auditoria de duplicatas e criação dos folds.

Entrada : raiz do projeto (<split>/<classe>/*.png, ex.: train/Dark/*.png)
Saídas  : outputs/audit/
            files.csv        todas as imagens, com md5 e grupo
            unique.csv       uma linha por imagem única (é a base do resto do pipeline)
            folds.json       15 splits (3 repetições x 5 folds) com índices de unique.csv
            report.json      números da auditoria (vão para o artigo)
            near_dups.png         pares quase-duplicados (se houver), para conferir a olho
            closest_pairs.png     pares de imagens distintas mais parecidas (margem do limiar)

Quase-duplicata = mesma foto com pequenas alterações (recompressão, redimensionamento).
Critério: RMSE pixel a pixel entre as imagens reduzidas para 64x64 RGB (escala 0-255).
O pHash foi descartado: como toda foto é "um grão centralizado num fundo claro", ele
dava distância <= 2 até para grãos diferentes de classes diferentes.

Mesmo grão fotografado de novo (girado, deslocado, com outra luz): detectado
automaticamente na etapa 3b (python -m src.same_bean) e gravado em
meta/mesmo_grao.csv. Os pares marcados com "s" na coluna mesmo_grao entram no
mesmo grupo dos folds.

Uso (na raiz do projeto): python -m src.audit
     --data padrão = raiz do projeto (onde estão train/ e test/)
     --out  padrão = <raiz>/outputs
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.model_selection import StratifiedGroupKFold

from .common import DEFAULT_OUT, ROOT, list_images, save_json


class UnionFind:
    """Junta índices em grupos (componentes conectados)."""

    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, i: int) -> int:
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, i: int, j: int) -> None:
        self.p[self.find(i)] = self.find(j)


def save_pairs_sheet(data_dir, df, pairs, path, size=160, max_pairs=24):
    """Prancha com pares (esquerda | direita | distância) para inspeção visual."""
    pairs = pairs[:max_pairs]
    if not pairs:
        return
    sheet = Image.new("RGB", (2 * size + 120, len(pairs) * size), "white")
    draw = ImageDraw.Draw(sheet)
    for r, (d, i, j) in enumerate(pairs):
        for c, k in enumerate((i, j)):
            im = Image.open(data_dir / df.path[k]).convert("RGB").resize((size - 4, size - 4))
            sheet.paste(im, (c * size + 2, r * size + 2))
        draw.text((2 * size + 6, r * size + 10), f"RMSE={d}", fill="black")
        draw.text((2 * size + 6, r * size + 30), df.label[i][:6] + " | " + df.label[j][:6], fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def pixel_rmse_matrix(data_dir: Path, paths, size: int = 64) -> np.ndarray:
    """RMSE (escala 0-255) entre todas as imagens, reduzidas para size x size RGB."""
    X = np.stack([np.asarray(Image.open(data_dir / p).convert("RGB")
                             .resize((size, size), Image.BILINEAR), dtype=np.float64).ravel()
                  for p in paths])
    sq = (X ** 2).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * X @ X.T
    return np.sqrt(np.maximum(d2, 0.0) / X.shape[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=ROOT, type=Path,
                    help="pasta com train/ e test/ (padrão: raiz do projeto)")
    ap.add_argument("--out", default=DEFAULT_OUT, type=Path)
    ap.add_argument("--pix-thr", type=float, default=10.0,
                    help="RMSE máximo (0-255, imagens 64x64) para considerar quase-duplicata. "
                         "No dataset completo, as mesmas fotos deram 3,0 e 6,9 e o par "
                         "distinto seguinte, 13,2")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--same-bean", default=ROOT / "meta" / "mesmo_grao.csv", type=Path,
                    help="pares do mesmo grão (gerado por src.same_bean)")
    args = ap.parse_args()
    out = args.out / "audit"
    out.mkdir(parents=True, exist_ok=True)

    # 1) inventário + hashes -------------------------------------------------
    df = list_images(args.data)
    print(f"[auditoria] {len(df)} imagens encontradas")
    df["md5"] = [hashlib.md5((args.data / p).read_bytes()).hexdigest() for p in df.path]

    # 2) duplicatas exatas (mesmo md5) --------------------------------------
    n_md5 = df.md5.nunique()
    by_md5 = df.groupby("md5")
    md5_labels = by_md5.label.nunique()
    conflicts = md5_labels[md5_labels > 1].index.tolist()  # mesma imagem, classes diferentes
    md5_splits = by_md5.split.agg(lambda s: set(s))
    test_rows = df[df.split == "test"]
    test_with_train_copy = int(sum("train" in md5_splits[m] for m in test_rows.md5))

    # 3) quase-duplicatas (RMSE de pixels) entre imagens com md5 diferente -
    dist = pixel_rmse_matrix(args.data, df.path)
    uf = UnionFind(len(df))
    for _, g in by_md5:              # cópias exatas: mesmo grupo
        idx = g.index.tolist()
        for k in idx[1:]:
            uf.union(idx[0], k)
    near_pairs = []
    iu, ju = np.triu_indices(len(df), 1)
    for i, j in zip(iu, ju):
        if df.md5[i] != df.md5[j] and dist[i, j] <= args.pix_thr:
            uf.union(i, j)
            near_pairs.append((round(float(dist[i, j]), 2), int(i), int(j)))
    near_pairs.sort()

    # 3b) mesmo grão em fotos diferentes (detectado por src.same_bean) ---------
    n_same_bean = 0
    if args.same_bean.exists():
        conf = pd.read_csv(args.same_bean, dtype=str).fillna("")
        conf = conf[conf.mesmo_grao.str.strip().str.lower() == "s"]
        rows_by_path = df.groupby("path").groups
        for a_path, b_path in zip(conf.path_a, conf.path_b):
            if a_path in rows_by_path and b_path in rows_by_path:
                uf.union(int(rows_by_path[a_path][0]), int(rows_by_path[b_path][0]))
                n_same_bean += 1
            else:
                print(f"[auditoria] aviso: par não encontrado no dataset: {a_path} | {b_path}")
    df["group"] = [uf.find(i) for i in range(len(df))]

    # pares distintos mais parecidos: mostram a margem entre "cópia" e "outro grão"
    distinct = df.md5.to_numpy()[iu] != df.md5.to_numpy()[ju]
    order = np.argsort(dist[iu, ju][distinct])[:24]
    closest = [(round(float(dist[iu[distinct][k], ju[distinct][k]]), 2),
                int(iu[distinct][k]), int(ju[distinct][k])) for k in order]
    min_distinct = closest[0][0] if closest else None
    save_pairs_sheet(args.data, df, near_pairs, out / "near_dups.png")
    save_pairs_sheet(args.data, df, closest, out / "closest_pairs.png")

    # 4) conjunto único: uma imagem por md5 (prefere a cópia de treino) ----
    df["split_rank"] = (df.split != "train").astype(int)
    uniq = (df.sort_values(["md5", "split_rank", "path"])
              .drop_duplicates("md5", keep="first")
              .sort_values(["y", "path"])
              .reset_index(drop=True))
    uniq = uniq[["path", "label", "y", "split", "md5", "group"]]
    # grupos com mais de uma imagem única = quase-duplicatas mantidas juntas
    uniq["group"] = pd.factorize(uniq.group)[0]

    # 5) folds: estratificados por classe, quase-duplicatas no mesmo lado --
    folds = []
    n_splits = min(args.folds, int(uniq.y.value_counts().min()))
    for rep in range(args.reps):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=rep)
        for k, (tr, te) in enumerate(sgkf.split(uniq, uniq.y, uniq.group)):
            folds.append({"rep": rep, "fold": k, "train": tr.tolist(), "test": te.tolist()})

    # 6) relatório ----------------------------------------------------------
    report = {
        "n_imagens": int(len(df)),
        "por_split": df.split.value_counts().to_dict(),
        "n_md5_distintos": int(n_md5),
        "n_copias_exatas_removidas": int(len(df) - n_md5),
        "teste_com_copia_exata_no_treino": test_with_train_copy,
        "n_teste": int(len(test_rows)),
        "md5_com_classes_conflitantes": len(conflicts),
        "quase_dup_criterio": "RMSE de pixels 64x64 RGB (0-255)",
        "quase_dup_limiar": args.pix_thr,
        "pares_quase_duplicados": len(near_pairs),
        "menor_rmse_entre_imagens_distintas": min_distinct,
        "pares_mesmo_grao_confirmados": n_same_bean,
        "n_unicas": int(len(uniq)),
        "unicas_por_classe": uniq.label.value_counts().to_dict(),
        "n_grupos": int(uniq.group.nunique()),
        "splits": f"{args.reps}x{n_splits} StratifiedGroupKFold",
    }
    df.drop(columns="split_rank").to_csv(out / "files.csv", index=False)
    uniq.to_csv(out / "unique.csv", index=False)
    save_json(folds, out / "folds.json")
    save_json(report, out / "report.json")

    print(f"[auditoria] md5 distintos: {n_md5} de {len(df)} "
          f"({len(df) - n_md5} cópias exatas)")
    print(f"[auditoria] imagens de teste com cópia exata no treino: "
          f"{test_with_train_copy} de {len(test_rows)}")
    print(f"[auditoria] quase-duplicatas (RMSE <= {args.pix_thr}): {len(near_pairs)} pares; "
          f"menor RMSE entre imagens distintas: {min_distinct}")
    if conflicts:
        print(f"[auditoria] ATENÇÃO: {len(conflicts)} imagens idênticas com classes diferentes")
    if n_same_bean:
        print(f"[auditoria] pares do mesmo grão confirmados (meta/mesmo_grao.csv): {n_same_bean}")
    print(f"[auditoria] conjunto único: {len(uniq)} imagens, {report['unicas_por_classe']}")
    print(f"[auditoria] {len(folds)} splits salvos em {out / 'folds.json'}")


if __name__ == "__main__":
    main()
