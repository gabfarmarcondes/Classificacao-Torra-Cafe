# Classificação de Torra de Café

Repositório para o artigo do Computer on the Beach com o *Coffee Bean Dataset Resized (224×224)*.

Pergunta de pesquisa: O que um classificador de nível de torra aprende a partir de imagens de grãos de café: as características do grão ou pistas externas a ele, como o fundo da foto, o formato e a origem do café?

## Estado

| Etapa | Script | Status |
|---|---|---|
| 0 · Auditoria de duplicatas + folds | `src/audit.py` | pronto |
| 1 · Máscara do grão + normalização pelo fundo | `src/preprocess.py` | pronto |
| 2 · Testes de atalho (fundo, forma, pares) | `src/shortcuts.py` | pronto |
| 3 · Extração (cor Lab + backbones) | — | próximo |
| 4–5 · Classificação e fusão | — | a fazer |
| 6 · Aumento e robustez | — | a fazer |
| 7 · Métricas e estatística | — | a fazer |
| 8 · t-SNE e Grad-CAM | — | a fazer |

## Estrutura

O código e o dataset ficam na mesma pasta:

```
coffe-bean-dataset/
├── src/              código do pipeline (audit, preprocess, shortcuts, ...)
├── train/            Green/ Light/ Medium/ Dark/  (300 imagens cada)  * não versionado
├── test/             Green/ Light/ Medium/ Dark/  (100 imagens cada)  * não versionado
├── Coffee Bean.csv   CSV original do Kaggle                           * não versionado
├── outputs/          criado pelo pipeline                             * não versionado
└── requirements.txt
```

### Dataset

O dataset não está no repositório. Baixe o [Coffee Bean Dataset Resized (224×224)](https://www.kaggle.com/datasets/gpiosenka/coffee-bean-dataset-resized-224-x-224) e coloque as pastas `train/` e `test/` na raiz do projeto, ao lado de `src/`.

## Instalação

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

O U²-Net (modelo de segmentação, ~176 MB) é baixado automaticamente na primeira execução da etapa 1.

## Como rodar

Rode na raiz do projeto. Como `train/` e `test/` estão ao lado de `src/`, `--data` e `--out` são opcionais (padrão: a própria raiz e `outputs/`).

```bash
python -m src.audit
python -m src.preprocess
python -m src.shortcuts
```

## O que cada etapa gera

### 0 · `outputs/audit/`
- `report.json`: números da auditoria (cópias exatas, teste com cópia no treino, quase-duplicatas, menor RMSE entre imagens distintas). **Vão para o artigo.**
- `unique.csv`: uma linha por imagem única. **É a base de todas as etapas seguintes.**
- `folds.json`: 15 splits (3 repetições × 5 folds, `StratifiedGroupKFold`). Os índices apontam para linhas de `unique.csv`.
- `files.csv`: todas as imagens, com md5 e grupo.
- `near_dups.png`: pares quase-duplicados, se houver, para conferir a olho.
- `closest_pairs.png`: os pares de imagens **distintas** mais parecidos. Mostram a margem entre "cópia" e "outro grão".

**Critério de quase-duplicata:** RMSE pixel a pixel entre as imagens reduzidas para 64×64 RGB (escala 0–255), com limiar 10 (no dataset completo, as fotos repetidas deram 3,0 e 6,9, e o par de grãos distintos seguinte, 13,2). O pHash foi testado e descartado: como toda foto é "um grão centralizado num fundo claro", ele dava distância ≤ 2 até para grãos diferentes de classes diferentes (Dark × Medium).

### 1 · `outputs/preprocess/`
- `masks.npz`: máscaras booleanas (N, 224, 224), na ordem de `unique.csv`.
- `norm/`: imagens normalizadas pelo fundo, com a mesma estrutura de pastas.
- `stats.csv`: fração de grão, cor do fundo e ganhos de cada imagem.
- `qc_<classe>.png`: contorno da máscara e imagem normalizada, para conferir.
- `suspeitas.csv`: só é criado se houver máscaras estranhas.

**Decisão de método:** a máscara usa o **U²-Net** (segmentador pré-treinado de objeto em destaque, via `rembg`). O limiar em Lab (croma/claridade) foi testado e funciona no Green e no Light, mas falha no Dark e no Medium: a sombra de contato é tão escura quanto o grão, e os reflexos são claros e sem cor como o fundo. Como a forma da máscara entra no teste de atalho, uma máscara com sombra criaria uma pista falsa (o tamanho da sombra depende da iluminação da sessão). O método por limiar continua disponível em `--method threshold`.

Para o artigo, cite o U²-Net: Qin et al., *U²-Net: Going deeper with nested U-structure for salient object detection*, Pattern Recognition, 2020.

### 2 · `outputs/shortcuts/` e `outputs/features/`
- `summary.csv`: acurácia e F1 (média ± desvio nos 15 splits) de cada teste. **Vai para o artigo.**
- `per_fold.csv`: resultado de cada split (para o Wilcoxon).
- `features/*.npy`: vetores de fundo, forma e cor por imagem, reaproveitados na etapa 3.

Testes (regressão logística):

| Teste | Usa | Acaso |
|---|---|---|
| `so_fundo` | 6 números do papel em volta do grão (original e normalizada) | 25% |
| `so_forma` | 11 números da silhueta, sem cor | 25% |
| `cor_grao` | 16 números da cor do grão: a pista legítima, como referência | 25% |
| `par_forma` | silhueta em Green×Light (mesma origem), Light×Medium e Medium×Dark (origens diferentes) | 50% |

## Resultados até a etapa 2 (dataset completo)

**Auditoria:** as 400 imagens do teste oficial são cópias exatas de imagens do treino. Ao todo há 406 cópias exatas; restam 1.194 imagens únicas (Green 299, Light 300, Medium 300, Dark 295). Há também 2 pares da mesma foto com mudança mínima (RMSE 3,0 e 6,9).

**Atalhos** (regressão logística, média ± desvio da acurácia nos 15 splits):

| Teste | Acurácia | Acaso |
|---|---|---|
| Só fundo (original) | 60,4% ± 3,1 | 25% |
| Só fundo (normalizado) | 53,9% ± 2,6 | 25% |
| Só forma | 49,0% ± 2,7 | 25% |
| Cor do grão (original) | 98,9% ± 0,7 | 25% |
| Cor do grão (normalizada) | 99,0% ± 0,5 | 25% |
| Forma: Green × Light (mesma origem) | 70,9% ± 4,7 | 50% |
| Forma: Light × Medium (origens diferentes) | 69,6% ± 4,0 | 50% |
| Forma: Medium × Dark (origens diferentes) | 63,6% ± 3,0 | 50% |
