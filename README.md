# Classificação de Torra de Café

Repositório para o artigo do Computer on the Beach com o *Coffee Bean Dataset Resized (224×224)*.

Pergunta de pesquisa: O que um classificador de nível de torra aprende a partir de imagens de grãos de café: as características do grão ou pistas externas a ele, como o fundo da foto, o formato e a origem do café?

## Estado

| Etapa | Script | Status |
|---|---|---|
| 0 · Auditoria de duplicatas + folds | `src/audit.py` | pronto |
| 1 · Máscara do grão + normalização pelo fundo | `src/preprocess.py` | pronto |
| 2 · Testes de atalho (fundo, forma, pares) | `src/shortcuts.py` | pronto |
| 3 · Extração com backbones (4 versões da imagem) | `src/deep_features.py` | pronto |
| 3b · Checagem de grão fotografado mais de uma vez | `src/same_bean.py` | pronto |
| 4–5 · Classificação e fusão | — | a fazer |
| 6 · Aumento e robustez | — | a fazer |
| 7 · Métricas e estatística | — | a fazer |
| 8 · t-SNE e Grad-CAM | — | a fazer |

## Estrutura

O código e o dataset ficam na mesma pasta:

```
coffe-bean-dataset/
├── src/              código do pipeline (audit, preprocess, shortcuts, ...)
├── meta/             anotações manuais versionadas (pares do mesmo grão)
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

O PyTorch é instalado à parte, na versão só para CPU (o pipeline não precisa de GPU):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Modelos baixados automaticamente na primeira execução: U²-Net (etapa 1), ResNet50 e ConvNeXt-T (torchvision) e DINOv2-S (Hugging Face) na etapa 3.

## Como rodar

Rode na raiz do projeto. Como `train/` e `test/` estão ao lado de `src/`, `--data` e `--out` são opcionais (padrão: a própria raiz e `outputs/`).

```bash
python -m src.audit
python -m src.preprocess
python -m src.shortcuts
python -m src.deep_features
python -m src.same_bean
```

Depois do `same_bean`, rode de novo `python -m src.audit` e `python -m src.shortcuts`: os pares do mesmo grão passam a ficar no mesmo grupo e os folds mudam.

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

### 3 · `outputs/features/deep_*`
- `deep_<backbone>_<versão>.npy`: uma matriz por backbone e versão da imagem (1.194 linhas, na ordem de `unique.csv`).
- `deep_meta.json`: dimensão e pesos de cada matriz.
- `versoes_exemplo.png`: as 4 versões lado a lado, uma imagem por classe.

Backbones congelados (só inferência, nenhum treino):

| Backbone | Pesos | Dimensão |
|---|---|---|
| `resnet50` | ImageNet (torchvision `IMAGENET1K_V2`) | 2048 |
| `convnext_tiny` | ImageNet (torchvision `IMAGENET1K_V1`) | 768 |
| `dinov2_small` | autossupervisionado (`facebook/dinov2-small`) | 384 |

Versões da imagem (`src/image_versions.py`):

| Versão | O que é | Para quê |
|---|---|---|
| `original` | a foto do dataset | referência |
| `normalizada` | iluminação corrigida pelo fundo (etapa 1) | testar a normalização |
| `sem_fundo` | grão original sobre branco puro | elimina a pista do fundo por construção |
| `so_fundo` | disco de raio fixo em volta do grão apagado e preenchido com o papel | controle: mede o quanto um backbone consegue usar só o fundo |

No `so_fundo`, o disco tem o mesmo tamanho em todas as imagens (cobre o maior grão do dataset, com margem), então nem o tamanho nem o formato do grão ficam visíveis. O preenchimento é uma superfície suave ajustada ao papel, para não criar manchas.

### 3b · `outputs/same_bean/` e `meta/mesmo_grao.csv`
A auditoria só detecta a mesma foto. O mesmo grão fotografado de novo (girado, deslocado, com outra luz) passa despercebido e pode cair no treino e no teste. A detecção é automática:

1. **Candidatos:** cada imagem `sem_fundo` passa pelo DINOv2 em 8 orientações. Para cada imagem, os 10 vizinhos mais parecidos da mesma classe viram pares candidatos.
2. **Verificação geométrica:** em cada par, pontos-chave SIFT detectados só dentro do grão são ligados entre as duas imagens, e o RANSAC conta quantas ligações são explicadas por uma mesma transformação (giro + deslocamento + zoom entre 0,6× e 1,6×). Mesmo grão = dezenas de ligações coerentes (rachaduras, manchas, borda); grãos diferentes = poucas, ao acaso.
3. **Limiar calibrado nos dados:** pares de classes diferentes são, com certeza, grãos diferentes. Os 5 vizinhos mais parecidos de outra classe de cada imagem são verificados do mesmo jeito; o limiar é o maior número de ligações entre eles + 1.
4. **Critério liberal, de propósito:** no dataset completo, as ligações dos pares da mesma classe formam uma faixa contínua, sem um vão entre "mesmo grão" e "grão diferente", e a inspeção visual não resolve o meio da faixa (alinhar duas fotos pela fenda faz até grãos diferentes parecerem iguais). Agrupar dois grãos diferentes por engano não prejudica nada (só os mantém do mesmo lado nos folds); deixar passar um grão repetido é vazamento. Por isso, tudo acima do que grãos distintos atingem por acaso é tratado como mesmo grão. O `resumo.json` registra quantos pares haveria com limiares mais rígidos.
5. Pares candidatos com ligações ≥ limiar são marcados como mesmo grão.

Saídas:
- `meta/mesmo_grao.csv` (versionado): uma linha por par candidato. Colunas: `similaridade` (DINOv2, usada só para escolher os candidatos), `ligacoes` (ligações SIFT coerentes entre as duas fotos) e `mesmo_grao` (`s` quando `ligacoes` ≥ limiar).
- `confirmados.png`: pares marcados, com as ligações desenhadas.
- `perto_do_limiar.png`: os rejeitados com mais ligações, para conferir a margem.
- `ligacoes.png`: distribuição das ligações (candidatos × classes diferentes).
- `resumo.json`: limiar, contagens e número de pares para limiares mais rígidos (sensibilidade). **Vai para o artigo.**

No dataset completo, os 5.376 pares de classes diferentes verificados deram no máximo 6 ligações.

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
