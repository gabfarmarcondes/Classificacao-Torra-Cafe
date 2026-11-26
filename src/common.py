"""Funções compartilhadas entre as etapas do pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage import color

# Raiz do projeto: contém src/, train/ e test/ (o dataset fica junto do código)
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "outputs"

# Ordem ordinal das classes: 0 = cru ... 3 = torra escura
CLASSES = ["Green", "Light", "Medium", "Dark"]
CLASS_TO_Y = {c: i for i, c in enumerate(CLASSES)}
IMG_EXTS = {".png", ".jpg", ".jpeg"}


# ---------------------------------------------------------------- arquivos
def list_images(data_dir: Path) -> pd.DataFrame:
    """Varre <data_dir>/<split>/<classe>/*.png e devolve uma tabela.

    Não depende do CSV do Kaggle, então funciona mesmo com subconjuntos.
    Olha só 3 níveis abaixo de data_dir: como o dataset fica na raiz do
    projeto, isso evita varrer venv/ e outputs/ (que tem cópias em norm/).
    """
    rows = []
    for p in sorted(data_dir.glob("*/*/*")):
        if p.suffix.lower() not in IMG_EXTS or not p.is_file():
            continue
        rel = p.relative_to(data_dir)
        split, label = rel.parts[0], rel.parts[1]
        if label not in CLASS_TO_Y:
            continue
        rows.append({"path": rel.as_posix(), "split": split, "label": label,
                     "y": CLASS_TO_Y[label]})
    if not rows:
        raise SystemExit(f"Nenhuma imagem encontrada em {data_dir} "
                         "(esperado: <split>/<classe>/*.png)")
    return pd.DataFrame(rows)


def load_rgb(path: Path) -> np.ndarray:
    """Carrega a imagem como uint8 (H, W, 3)."""
    return np.asarray(Image.open(path).convert("RGB"))


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def load_json(path: Path):
    return json.loads(path.read_text())


# ---------------------------------------------------------------- cor
def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB -> Lab (L* em 0..100, a* e b* aprox. -128..127)."""
    return color.rgb2lab(rgb.astype(np.float64) / 255.0)


def chroma(lab: np.ndarray) -> np.ndarray:
    """Croma = distância até o cinza = sqrt(a*^2 + b*^2)."""
    return np.hypot(lab[..., 1], lab[..., 2])


# ---------------------------------------------------------------- folds
def load_folds(out_dir: Path) -> list[dict]:
    """Lista de splits: {'rep', 'fold', 'train', 'test'} (índices em unique.csv)."""
    return load_json(out_dir / "audit" / "folds.json")


def load_unique(out_dir: Path) -> pd.DataFrame:
    return pd.read_csv(out_dir / "audit" / "unique.csv")
