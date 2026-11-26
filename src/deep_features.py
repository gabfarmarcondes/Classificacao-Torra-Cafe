"""Etapa 3 - Extração de features com backbones pré-treinados congelados.

Entrada : outputs/audit/unique.csv, outputs/preprocess/ (máscaras e imagens normalizadas)
Saídas  : outputs/features/deep_<backbone>_<versão>.npy   matriz (N imagens x D), float32,
                                                          na ordem de unique.csv
          outputs/features/deep_meta.json                 dimensões e pesos usados
          outputs/features/versoes_exemplo.png            as 4 versões lado a lado

Backbones (só inferência, nenhum treino):
  resnet50       torchvision, pesos ImageNet (IMAGENET1K_V2)  -> 2048 (média global da última camada conv)
  convnext_tiny  torchvision, pesos ImageNet (IMAGENET1K_V1)  -> 768  (antes da camada de classificação)
  dinov2_small   Hugging Face facebook/dinov2-small (autossupervisionado) -> 384 (token CLS)

Versões da imagem: ver src/image_versions.py (original, normalizada, sem_fundo, so_fundo).

Os pesos são baixados na primeira execução. Matrizes já existentes são puladas
(use --overwrite para refazer).

Uso (na raiz do projeto): python -m src.deep_features
     python -m src.deep_features --backbones dinov2_small --versions sem_fundo
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .common import DEFAULT_OUT, ROOT, load_unique, save_json
from .image_versions import VERSIONS, erase_radius, load_version

BACKBONES = ["resnet50", "convnext_tiny", "dinov2_small"]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Backbone(torch.nn.Module):
    """Envolve cada rede para devolver um vetor por imagem."""

    def __init__(self, name: str, pretrained: bool = True):
        super().__init__()
        self.name = name
        if name == "resnet50":
            from torchvision.models import ResNet50_Weights, resnet50
            self.net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            self.net.fc = torch.nn.Identity()
            self.dim, self.weights = 2048, "torchvision ResNet50_Weights.IMAGENET1K_V2"
        elif name == "convnext_tiny":
            from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
            self.net = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
            self.net.classifier[2] = torch.nn.Identity()
            self.dim, self.weights = 768, "torchvision ConvNeXt_Tiny_Weights.IMAGENET1K_V1"
        elif name == "dinov2_small":
            from transformers import Dinov2Config, Dinov2Model
            if pretrained:
                self.net = Dinov2Model.from_pretrained("facebook/dinov2-small")
            else:  # mesma arquitetura, pesos aleatórios (só para testar o código)
                self.net = Dinov2Model(Dinov2Config(hidden_size=384, num_hidden_layers=12,
                                                    num_attention_heads=6, intermediate_size=1536,
                                                    patch_size=14, image_size=518))
            self.dim, self.weights = 384, "huggingface facebook/dinov2-small"
        else:
            raise ValueError(f"backbone desconhecido: {name} (use um de {BACKBONES})")
        if not pretrained:
            self.weights = "ALEATÓRIOS (teste de código)"
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.name == "dinov2_small":
            return self.net(pixel_values=x).pooler_output   # token CLS após layernorm
        return self.net(x)


def to_tensor(images: list[np.ndarray], size: int = 224) -> torch.Tensor:
    """Lista de uint8 HxWx3 -> tensor normalizado com média/desvio do ImageNet."""
    arr = []
    for im in images:
        if im.shape[0] != size or im.shape[1] != size:
            im = np.asarray(Image.fromarray(im).resize((size, size), Image.BILINEAR))
        arr.append(im)
    x = torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2).float() / 255.0
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def extract(model: Backbone, images_iter, n: int, batch: int, device: str) -> np.ndarray:
    """Passa as imagens pela rede em lotes e devolve (n, dim)."""
    feats, buf = [], []
    for im in images_iter:
        buf.append(im)
        if len(buf) == batch:
            feats.append(model(to_tensor(buf).to(device)).cpu().numpy())
            buf = []
    if buf:
        feats.append(model(to_tensor(buf).to(device)).cpu().numpy())
    out = np.concatenate(feats).astype(np.float32)
    assert out.shape == (n, model.dim), out.shape
    return out


def save_versions_example(uniq, masks, data_dir, out_dir, path, radius, size=150):
    """Uma imagem por classe, nas 4 versões (colunas)."""
    rows = [uniq.index[uniq.label == c][0] for c in ["Green", "Light", "Medium", "Dark"]
            if (uniq.label == c).any()]
    sheet = Image.new("RGB", (len(VERSIONS) * size, len(rows) * size), "white")
    for r, i in enumerate(rows):
        for c, v in enumerate(VERSIONS):
            im = load_version(v, uniq.path[i], masks[i], data_dir, out_dir, radius)
            sheet.paste(Image.fromarray(im).resize((size - 4, size - 4)), (c * size + 2, r * size + 2))
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=ROOT, type=Path,
                    help="pasta com train/ e test/ (padrão: raiz do projeto)")
    ap.add_argument("--out", default=DEFAULT_OUT, type=Path)
    ap.add_argument("--backbones", nargs="+", default=BACKBONES, choices=BACKBONES)
    ap.add_argument("--versions", nargs="+", default=VERSIONS, choices=VERSIONS)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--random-weights", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    fdir = args.out / "features"
    fdir.mkdir(parents=True, exist_ok=True)
    uniq = load_unique(args.out)
    masks = np.load(args.out / "preprocess" / "masks.npz")["masks"]
    if len(masks) != len(uniq):
        raise SystemExit("máscaras e unique.csv com tamanhos diferentes: rode a etapa 1 de novo")
    radius = erase_radius(masks)
    save_versions_example(uniq, masks, args.data, args.out, fdir / "versoes_exemplo.png", radius)

    meta_path = fdir / "deep_meta.json"
    meta = {}
    if meta_path.exists():
        import json
        meta = json.loads(meta_path.read_text())

    print(f"[extração] {len(uniq)} imagens | dispositivo: {args.device} | "
          f"raio do disco em so_fundo: {radius} px")
    for bname in args.backbones:
        todo = [v for v in args.versions
                if args.overwrite or not (fdir / f"deep_{bname}_{v}.npy").exists()]
        if not todo:
            print(f"[extração] {bname}: todas as versões já existem, pulando")
            continue
        model = Backbone(bname, pretrained=not args.random_weights).to(args.device)
        for v in todo:
            imgs = (load_version(v, p, masks[i], args.data, args.out, radius)
                    for i, p in enumerate(uniq.path))
            X = extract(model, imgs, len(uniq), args.batch, args.device)
            np.save(fdir / f"deep_{bname}_{v}.npy", X)
            meta[f"deep_{bname}_{v}"] = {"backbone": bname, "versao": v,
                                         "dim": model.dim, "pesos": model.weights}
            if v == "so_fundo":
                meta[f"deep_{bname}_{v}"]["raio_disco_px"] = radius
            print(f"[extração] {bname} / {v}: {X.shape}")
        del model
    save_json(meta, meta_path)
    print(f"[extração] matrizes em {fdir}")


if __name__ == "__main__":
    main()
